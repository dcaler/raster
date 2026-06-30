"""Unit tests for the generalized doer: prompt building, file parse/write guards,
pytest helpers, queue linearization, and build/test dry-runs — all offline (no
Ollama, no trundlr, no git)."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raster import execlib
from raster.build import resolve_rung, rung_index, start_index
from raster.config import Config
from raster.cli import main
from raster.queue import linearize
from raster.spec import DEFAULT_LADDER, Project, find_gate, find_task

SPEC = {
    "meta": {"package": "pkg", "project": "P",
             "workers": {"strong": "qwen", "worker": "llama"}, "think": False},
    "execution": {"resources": {"gpu": 2, "cpu": 3}, "ollama_host": "http://localhost:11434"},
    "modules": [
        {"id": "P0", "name": "test-infra", "tasks": [
            {"id": "P0.M0", "title": "Freeze M0 tests", "worker": "strong",
             "deliverables": ["tests/test_smoke.py"], "spec": "author frozen tests",
             "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest --collect-only -q tests/"}},
        ]},
        {"id": "M0", "name": "scaffold", "tasks": [
            {"id": "M0.T1", "title": "Package scaffold", "worker": "worker",
             "deliverables": ["pkg/__init__.py", "tests/test_smoke.py"], "spec": "make it import",
             "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest -q tests/test_smoke.py"}},
        ], "gate": {"id": "G0", "spec": "package imports",
                    "integration_test": {"file": "tests/gate.py", "cmd": "pytest -q tests/gate.py"}}},
    ],
}


def make_project(tmp_path) -> Project:
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    return Project(root=tmp_path, code=code, cfg=Config(),
                   ry={"project": "P", "package": "pkg"}, spec=SPEC)


# ------------------------------------------------------------------- execlib units
def test_parse_files():
    text = ("preamble\n=== FILE: pkg/a.py ===\nx = 1\n=== END FILE ===\n"
            "=== FILE: pkg/b.py ===\ny = 2\n=== END FILE ===\n")
    files = execlib.parse_files(text)
    assert files == {"pkg/a.py": "x = 1", "pkg/b.py": "y = 2"}


def test_parse_diagnostics_unterminated():
    # one closed block + one opened-but-not-closed (missing === END FILE ===)
    text = ("=== FILE: pkg/a.py ===\nx = 1\n=== END FILE ===\n"
            "=== FILE: pkg/run.py ===\nimport sys\nprint('no closer here')\n")
    files = execlib.parse_files(text)
    assert files == {"pkg/a.py": "x = 1"}                  # strict parse drops the unterminated one
    diag = execlib.parse_diagnostics(text, files)
    assert diag["parsed"] == ["pkg/a.py"]
    assert diag["unterminated"] == ["pkg/run.py"]          # the dropped file is surfaced
    # self-diagnosing reason (V) and targeted re-prompt (U) both name the unterminated path
    assert "no `=== END FILE ===`" in execlib.parse_failure_reason(diag)
    rp = execlib.reprompt_for_parse_failure(diag)
    assert "pkg/run.py" in rp and "=== END FILE ===" in rp


def test_parse_diagnostics_no_marker():
    diag = execlib.parse_diagnostics("just prose, no file blocks", {})
    assert diag["opened"] == [] and diag["unterminated"] == []
    assert "no `=== FILE:`" in execlib.parse_failure_reason(diag)
    # targeted re-prompt names the root so the worker doesn't re-prefix it
    rp = execlib.reprompt_for_parse_failure(diag, root="code")
    assert "=== FILE:" in rp and "code/" in rp


def test_parse_diagnostics_clean():
    text = ("=== FILE: pkg/a.py ===\nx = 1\n=== END FILE ===\n"
            "=== FILE: pkg/b.py ===\ny = 2\n=== END FILE ===\n")
    diag = execlib.parse_diagnostics(text, execlib.parse_files(text))
    assert diag["unterminated"] == []                      # nothing dropped -> no warning fires


def _drive_build(tmp_path, monkeypatch, run_test_output):
    """Drive the real run_build loop offline. `run_test_output(attempt) -> output` lets a test
    choose whether successive failures are identical (plateau) or distinct. Returns (rc, sizes)
    where sizes[i] is the total prompt chars fed to the model on attempt i+1."""
    from types import SimpleNamespace
    from raster import build, ollama
    project = make_project(tmp_path)
    monkeypatch.setattr(build, "load_project", lambda d: project)
    sizes, n = [], {"i": 0}

    def fake_chat(host, model, messages, label="", think=False):
        sizes.append(sum(len(m["content"]) for m in messages))
        return "=== FILE: pkg/__init__.py ===\nx = 1\n=== END FILE ===\n"   # well-formed, will FAIL

    def fake_run_test(proj, cmd, stub_pkg=None, timeout=None):
        n["i"] += 1
        return False, run_test_output(n["i"])

    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(execlib, "run_test", fake_run_test)
    monkeypatch.setattr(execlib, "git_commit_push", lambda *a, **k: None)
    args = SimpleNamespace(dir=str(tmp_path), task="M0.T1", dry_run=False, max_attempts=4)
    return build.run_build(args), sizes


def test_build_loop_recomposes_prompt_not_accumulates(tmp_path, monkeypatch):
    # DISTINCT failure each attempt -> no plateau -> runs the full budget. The prompt is
    # RE-COMPOSED each attempt (task + contract + on-disk code + ONE failure summary), so retry
    # prompts are the same shape, not an ever-growing transcript (X/Y).
    rc, sizes = _drive_build(tmp_path, monkeypatch,
                             lambda i: f"E   assert {i} == 99\nFAILED tests/test_smoke.py::test_x")
    assert rc == 1 and len(sizes) == 4                     # all 4 attempts ran (failures differ)
    assert sizes[1] == sizes[2] == sizes[3]                # every retry prompt is the same size
    assert sizes[3] < sizes[1] * 1.5                       # no runaway accumulation across the loop


def test_build_loop_aborts_on_plateau_before_escalating(tmp_path, monkeypatch):
    # IDENTICAL failure each attempt -> byte-identical signature on attempt 2 -> abort to an
    # oracle check WITHOUT spending the strong tier (ESCALATE_AFTER=2 -> attempt 3 would escalate).
    rc, sizes = _drive_build(tmp_path, monkeypatch,
                             lambda i: "E   assert 0.0833 == 0.8\nFAILED tests/test_smoke.py::test_x")
    assert rc == 1
    assert len(sizes) == 2                                 # aborted on the 2nd identical failure


def test_build_loop_moving_chain_recommends_requeue(tmp_path, monkeypatch, capsys):
    # A CHANGING failure each attempt (a moving chain, NOT a plateau) runs the full budget; raster
    # logs the changed signature as PROGRESS mid-loop (EE) and the final log flags a re-queue, the
    # mirror image of the plateau's reconcile (GG). The driver writes clean code, so no undef pass.
    rc, sizes = _drive_build(tmp_path, monkeypatch,
                             lambda i: f"E   assert {i} == 99\nFAILED tests/test_smoke.py::test_x")
    assert rc == 1 and len(sizes) == 4
    out = capsys.readouterr().out
    assert "signature CHANGED" in out                      # EE diagnostic fires mid-loop
    assert "MOVING chain" in out and "RE-QUEUE" in out      # GG: more attempts, not a reconcile


def test_failed_count_reads_pytest_summary():
    # the trajectory the loop reads: a count when failures exist, 0 when none/unparseable (OO).
    assert execlib.failed_count("17 failed, 17 passed in 4.2s") == 17
    assert execlib.failed_count("E   NameError\nFAILED tests/test_x.py::test_y") == 0
    assert execlib.failed_count("") == 0


def test_build_loop_decaying_plateau_recommends_reconcile(tmp_path, monkeypatch, capsys):
    # A CHANGING signature (so NOT the byte-identical plateau) whose FAILED COUNT falls then LEVELS
    # OFF above zero (19->18->17->17) is a DECAYING PLATEAU: the worker fixed everything satisfiable
    # and hit a floor of unsatisfiable oracle bugs (M9.T1). raster reads the asymptote, not the slope
    # (OO/PP): it runs the budget but the verdict flips from "re-queue" to RECONCILE.
    counts = {1: 19, 2: 18, 3: 17, 4: 17}
    rc, sizes = _drive_build(
        tmp_path, monkeypatch,
        lambda i: f"E   assert {i} == 99\nFAILED tests/test_x.py::test_case_{i}\n"
                  f"{counts[i]} failed, 5 passed in 4.0s")
    assert rc == 1 and len(sizes) == 4                     # changing signature -> full budget, no abort
    out = capsys.readouterr().out
    assert "DECAYING PLATEAU" in out                       # the floor is named, not "progress"
    assert "RECONCILE" in out and "RE-QUEUE" not in out     # reconcile the residual, do NOT add turns


def test_build_loop_descending_chain_reads_asymptote(tmp_path, monkeypatch, capsys):
    # The companion: a changing chain whose count is STILL falling (19->18->17->16) re-queues, but
    # raster now flags the asymptote check first — confirm it heads to ZERO, not a nonzero floor.
    counts = {1: 19, 2: 18, 3: 17, 4: 16}
    rc, sizes = _drive_build(
        tmp_path, monkeypatch,
        lambda i: f"E   assert {i} == 99\nFAILED tests/test_x.py::test_case_{i}\n"
                  f"{counts[i]} failed, 5 passed in 4.0s")
    assert rc == 1 and len(sizes) == 4
    out = capsys.readouterr().out
    assert "still DESCENDING" in out and "READ THE ASYMPTOTE" in out
    assert "RE-QUEUE" in out                               # still the re-queue lever, with the caveat


# ---------------------------------------------------- undefined-name pass (FF) + reprompt
def test_undefined_names_detects(tmp_path):
    project = make_project(tmp_path)
    (project.code / "pkg" / "__init__.py").write_text("")
    (project.code / "pkg" / "m.py").write_text(
        "import os\n"
        "def f(a):\n"
        "    return os.path.join(a, missing(b))\n")        # `missing` and `b` unbound; `a`,`os` bound
    found = {name for _, _, name in execlib.undefined_names(project)}
    assert found == {"missing", "b"}


def test_undefined_names_clean_and_bails_on_star(tmp_path):
    project = make_project(tmp_path)
    (project.code / "pkg" / "__init__.py").write_text("")
    # walrus, comprehension target, and except-as all count as BOUND -> zero false positives
    (project.code / "pkg" / "clean.py").write_text(
        "def g(items):\n"
        "    try:\n"
        "        n = sum((y := x) for x in items)\n"
        "    except ValueError as e:\n"
        "        n = repr(e)\n"
        "    return n\n")
    assert execlib.undefined_names(project) == []
    # a star import hides its bindings -> bail on that file rather than risk a false finding
    (project.code / "pkg" / "star.py").write_text("from os import *\nv = getcwd()\nq = whatever\n")
    assert all(rel != "pkg/star.py" for rel, _, _ in execlib.undefined_names(project))


def test_undefined_names_noop_when_unbuilt(tmp_path):
    project = make_project(tmp_path)                        # pkg/ exists but empty
    assert execlib.undefined_names(project) == []


def test_reprompt_for_undefined_names_lists_all():
    note = execlib.reprompt_for_undefined_names(
        [("pkg/m.py", 3, "Path"), ("pkg/m.py", 5, "jaccard_distance")])
    assert "Path" in note and "jaccard_distance" in note   # the WHOLE list, not just the first
    assert "ALL" in note                                   # instruct fixing every one in one pass


def test_build_loop_surfaces_undefined_names_in_feedback(tmp_path, monkeypatch):
    # FF: the worker emits product code with undefined names; the static pass surfaces them and the
    # loop folds the FULL list into the next attempt's feedback (one repair clears the chain).
    from types import SimpleNamespace
    from raster import build, ollama
    project = make_project(tmp_path)
    monkeypatch.setattr(build, "load_project", lambda d: project)
    seen_feedback = []

    def fake_chat(host, model, messages, label="", think=False):
        seen_feedback.append("\n".join(m["content"] for m in messages[1:]))   # the feedback msg(s)
        return ("=== FILE: pkg/__init__.py ===\n"
                "import os\nval = Path(os.getcwd()) / missing_helper()\n"
                "=== END FILE ===\n")

    def fake_run_test(proj, cmd, stub_pkg=None, timeout=None):
        return False, "E   NameError: name 'Path' is not defined\nFAILED tests/test_smoke.py::test_x"

    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(execlib, "run_test", fake_run_test)
    monkeypatch.setattr(execlib, "git_commit_push", lambda *a, **k: None)
    args = SimpleNamespace(dir=str(tmp_path), task="M0.T1", dry_run=False, max_attempts=2)
    assert build.run_build(args) == 1
    # attempt 2's feedback names BOTH undefined names at once — not just the first NameError pytest
    # reported (`Path`) but also the masked `missing_helper` the interpreter never reached.
    assert "Path" in seen_feedback[1] and "missing_helper" in seen_feedback[1]


# ------------------------------------------- context sizing (LL/NN) + prompt trim (MM)
def test_pick_num_ctx_scales_and_clamps():
    from raster.ollama import CHARS_PER_TOKEN, MAX_NUM_CTX, MIN_NUM_CTX, pick_num_ctx
    # a tiny prompt floors at MIN; the window is always a power of two
    assert pick_num_ctx(10, output_tokens=0) == MIN_NUM_CTX
    # a prompt of ~2*MIN tokens lands in the 2*MIN bucket — rounded UP, still a power of two (NN)
    chars = MIN_NUM_CTX * 2 * CHARS_PER_TOKEN
    ctx = pick_num_ctx(chars, output_tokens=0)
    assert ctx == MIN_NUM_CTX * 2 and (ctx & (ctx - 1)) == 0
    # output headroom counts toward the window BEFORE bucketing -> can bump it up a notch (NN: the
    # reply shares the context, so size for it too)
    assert pick_num_ctx(chars, output_tokens=MIN_NUM_CTX * 2) > ctx
    # clamped to MAX no matter how huge the prompt (the caller logs the truncation risk instead)
    assert pick_num_ctx(MAX_NUM_CTX * 100, output_tokens=0) == MAX_NUM_CTX


def test_estimate_tokens_ceils():
    from raster.ollama import CHARS_PER_TOKEN, estimate_tokens
    assert estimate_tokens(0) == 0
    assert estimate_tokens(1) == 1                          # ceil: any chars -> at least 1 token
    assert estimate_tokens(CHARS_PER_TOKEN * 3) == 3
    assert estimate_tokens(CHARS_PER_TOKEN * 3 + 1) == 4    # rounds up, never truncates the estimate


def test_api_digest_keeps_signatures_drops_bodies():
    src = (
        "import os\n"
        "from math import sqrt\n"
        "TABLE = {'a': 1}\n"
        "DELTA: float = 0.5\n"
        "def helper(x, y=3):\n"
        '    """Combine x and y.\n\n    Extra detail line.\n    """\n'
        "    secret_body_token = x + y\n"
        "    return secret_body_token\n"
        "class Widget:\n"
        '    """A widget."""\n'
        "    def render(self, n):\n"
        "        return n * 2\n"
    )
    d = execlib.api_digest(src)
    assert "import os" in d and "from math import sqrt" in d   # imports kept (deps the worker calls)
    assert "def helper(x, y=3)" in d                           # function signature kept
    assert "Combine x and y." in d and "Extra detail line." not in d   # only the FIRST docstring line
    assert "class Widget" in d and "def render(self, n)" in d  # class + method signatures kept
    assert "TABLE = ..." in d and "DELTA: float" in d          # module-level constant NAMES kept
    assert "secret_body_token" not in d                        # bodies dropped (the whole point)


def test_api_digest_fallback_on_syntax_error():
    # unparseable source is returned in FULL — never hide existing code over a parse error
    bad = "def broken(:\n    pass\n"
    assert execlib.api_digest(bad) == bad


def test_package_api_digest_full_body_only_for_edited(tmp_path):
    project = make_project(tmp_path)
    pkg = project.code / "pkg"
    (pkg / "__init__.py").write_text("")
    (pkg / "helpers.py").write_text(
        "def existing_helper(a):\n    helper_only_marker = a + 1\n    return helper_only_marker\n")
    (pkg / "target.py").write_text(
        "def edit_me():\n    target_full_marker = 1\n    return target_full_marker\n")
    out = execlib.package_api_digest(project, full_bodies={"pkg/target.py"})
    # the edited file is shown in FULL (body visible); every other module is signature-only
    assert "=== EXISTING (full): pkg/target.py ===" in out and "target_full_marker" in out
    assert "=== API: pkg/helpers.py ===" in out and "def existing_helper(a)" in out
    assert "helper_only_marker" not in out                     # non-edited module body trimmed away


def test_build_prompt_trims_non_edited_modules(tmp_path):
    project = make_project(tmp_path)
    pkg = project.code / "pkg"
    (pkg / "__init__.py").write_text("")
    (pkg / "other.py").write_text(
        "def far_away():\n    deep_impl_marker = 7\n    return deep_impl_marker\n")
    module, task = find_task(SPEC, "M0.T1")   # edits pkg/__init__.py, NOT pkg/other.py
    prompt = execlib.build_prompt(project, module, task, authoring=False)
    assert "Existing package API" in prompt
    assert "def far_away()" in prompt          # signature offered so the worker can call it
    assert "deep_impl_marker" not in prompt    # but its body is trimmed (context-sizing, MM)


def test_write_files_guards(tmp_path):
    project = make_project(tmp_path)
    files = {"pkg/a.py": "x = 1", "tests/t.py": "frozen", "../escape.py": "nope"}
    written = execlib.write_files(project, files, allow_tests=False)
    assert written == ["pkg/a.py"]                      # tests/ refused, escape refused
    assert (project.code / "pkg" / "a.py").read_text() == "x = 1\n"
    assert not (project.code / "tests").exists()
    assert not (tmp_path / "escape.py").exists()
    # authoring may write tests/
    assert execlib.write_files(project, {"tests/t.py": "frozen"}, allow_tests=True) == ["tests/t.py"]


def test_write_files_single_owner_protection(tmp_path):
    project = make_project(tmp_path)
    owners = {"tests/conftest.py": "P0.T0"}
    # a non-owner authoring task is refused (would clobber the owner's shared infra)
    assert execlib.write_files(project, {"tests/conftest.py": "x"}, allow_tests=True,
                               owners=owners, task_id="P0.M1") == []
    assert not (project.code / "tests" / "conftest.py").exists()
    # the owning task may write it
    assert execlib.write_files(project, {"tests/conftest.py": "y"}, allow_tests=True,
                               owners=owners, task_id="P0.T0") == ["tests/conftest.py"]


def test_authoring_owners_and_owner_of():
    from raster.spec import authoring_owners, owner_of
    spec = {"modules": [
        {"id": "P0", "tasks": [
            {"id": "P0.T0", "deliverables": ["tests/conftest.py", "tests/golden/"]},
            {"id": "P0.M1", "deliverables": ["tests/test_grid.py", "tests/conftest.py"]},
        ]},
        {"id": "M1", "tasks": [{"id": "M1.T1", "deliverables": ["pkg/grid.py"]}]},
    ]}
    owners = authoring_owners(spec)
    assert owners["tests/conftest.py"] == "P0.T0"        # first P0 declarer owns it
    assert owners["tests/test_grid.py"] == "P0.M1"
    assert "pkg/grid.py" not in owners                   # only P0.* author shared infra
    assert owner_of(owners, "tests/golden/consts.py") == "P0.T0"   # dir deliverable owns subtree
    assert owner_of(owners, "tests/test_new.py") is None          # unowned -> writable


def test_freeze_stub_resolves_absent_product():
    import importlib
    from raster._freezestub import _StubFinder
    finder = _StubFinder("notreal_product_xyz")
    sys.meta_path.append(finder)
    try:
        cfg = importlib.import_module("notreal_product_xyz.config")
        assert cfg.AnyName is not None                   # any attribute resolves to a dummy
        from notreal_product_xyz.model import Whatever   # noqa: F401 — submodule + name resolve
        assert callable(Whatever)                        # dummy is callable: inert at collect time
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.startswith("notreal_product_xyz")]:
            del sys.modules[k]


def test_freezelint_catches_cross_reference_defects(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    (tests / "golden.py").write_text('DIATONIC = {"C": 0, "G": 7, "F": 5}\n')
    (tests / "conftest.py").write_text(
        "import pytest\n@pytest.fixture\ndef real_fixture():\n    return 1\n")
    (tests / "test_a.py").write_text(
        "import pytest\n"
        "from golden import DIATONIC\n"
        "from pkg.model import Model\n"
        "def test_key_ok(real_fixture, tmp_path):\n    assert DIATONIC['C'] == 0\n"
        "def test_key_bad():\n    assert DIATONIC['I'] == 0\n"               # unresolvable golden key
        "def test_missing(ghost_fixture):\n    assert ghost_fixture\n"       # undefined fixture
        "@pytest.mark.parametrize('n,expected', [(1, 1)])\n"
        "def test_param(n, expected):\n    assert n == expected\n"           # params are NOT fixtures
        "def test_pos():\n    Model(some_config)\n")                         # positional call
    (tests / "test_b.py").write_text(
        "from pkg.model import Model\n"
        "def test_kw():\n    Model(n_chord_types=3, bars_per_window=4)\n")   # keyword call -> schism

    v = "\n".join(lint_frozen_tests(code, "pkg"))
    assert "DIATONIC['I']" in v and "key absent" in v          # golden-key resolvability
    assert "ghost_fixture" in v and "defined nowhere" in v     # fixture resolvability
    assert "real_fixture" not in v and "tmp_path" not in v     # defined + builtin -> not flagged
    assert "'expected'" not in v and "requests fixture 'n'" not in v   # parametrize names excluded
    assert "Model called inconsistently" in v                 # call-signature coherence


def test_freezelint_skip_on_importerror_and_module_schism(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    (tests / "test_voc.py").write_text(
        "import pytest\n"
        "try:\n"
        "    from pkg import chord\n"                    # schism: deliverable is pkg/chords.py
        "except ImportError:\n"
        "    pytest.skip('not yet implemented', allow_module_level=True)\n"
        "import pkg.chord\n"                             # unambiguous schism import
        "def test_x():\n    assert chord\n")
    spec = {"meta": {"package": "pkg"}, "modules": [
        {"id": "M1", "tasks": [{"id": "M1.T1", "deliverables": ["pkg/chords.py"]}]}]}
    v = "\n".join(lint_frozen_tests(code, "pkg", spec))
    assert "skip-on-ImportError" in v                    # idiom flagged (check A)
    assert "no task declares" in v and "pkg.chord" in v  # module-import resolvability (check B)
    assert "did you mean pkg.chords?" in v               # singular/plural hint


def test_freezelint_module_resolvability_clean(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    (code / "tests").mkdir(parents=True)
    (code / "tests" / "test_ok.py").write_text(
        "import pkg.chords\nfrom pkg.metrics import jaccard\ndef test_x():\n    assert pkg.chords\n")
    spec = {"meta": {"package": "pkg"}, "modules": [{"id": "M1", "tasks": [
        {"id": "M1.T1", "deliverables": ["pkg/chords.py", "pkg/metrics/__init__.py"]}]}]}
    assert lint_frozen_tests(code, "pkg", spec) == []    # both modules declared -> clean


def test_declared_modules():
    from raster.spec import declared_modules
    spec = {"modules": [{"id": "M1", "tasks": [
        {"id": "M1.T1", "deliverables": ["pkg/chords.py", "pkg/metrics/__init__.py",
                                         "pkg/__init__.py", "tests/test_x.py", "pyproject.toml"]}]}]}
    assert declared_modules(spec, "pkg") == {"pkg.chords", "pkg.metrics", "pkg"}


def test_skipped_count():
    assert execlib.skipped_count("=== 50 passed, 3 skipped in 1.2s ===") == 3
    assert execlib.skipped_count("=== 50 passed in 1.2s ===") == 0


def test_run_test_honors_per_task_budget(tmp_path, monkeypatch):
    # a per-task `budget:` (seconds) overrides the global TEST_TIMEOUT so a legitimately long
    # gate (a GA over seeded sim runs) isn't killed and misread as a failure.
    import subprocess
    from raster.execlib import TEST_TIMEOUT
    project = make_project(tmp_path)
    seen = {}

    class _Done:
        returncode, stdout, stderr = 0, "1 passed", ""

    def fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    execlib.run_test(project, "pytest -q", timeout=1800)
    assert seen["timeout"] == 1800                         # the override is used
    execlib.run_test(project, "pytest -q")                 # no override -> the global default
    assert seen["timeout"] == TEST_TIMEOUT
    execlib.run_test(project, "pytest -q", timeout=None)   # falsy budget -> global default, not None
    assert seen["timeout"] == TEST_TIMEOUT


def test_freezelint_phantom_attr_spy_sweeps_whole_tree(tmp_path):
    from raster.freezelint import lint_phantom_attr_spies
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / "tests").mkdir()
    # the built product increments `steps` (Mesa) and exposes `beat_index` — but never `_step_count`
    (code / "pkg" / "model.py").write_text(
        "class Model:\n    def __init__(self):\n        self.steps = 0\n        self.beat_index = 0\n")
    # the SAME wrong belief sits in the unit test AND in the gate (a separate file) — RR/SS
    (code / "tests" / "test_player.py").write_text(
        "from pkg.model import Model\n"
        "def test_step():\n    m = Model()\n"
        "    assert getattr(m, '_step_count', 0) == 1\n"          # phantom: product has no _step_count
        "    assert getattr(m, 'beat_index', 0) == 0\n")          # real attr -> NOT flagged
    (code / "tests" / "gate_gui.py").write_text(
        "from pkg.model import Model\n"
        "def test_gate():\n    m = Model()\n"
        "    assert getattr(m, '_step_count', 0) == 2\n")          # the gate clone the unit green misses
    v = lint_phantom_attr_spies(code, "pkg")
    joined = "\n".join(v)
    assert sum("_step_count" in x for x in v) == 2                 # flagged in BOTH files (the sweep)
    assert "gate_gui.py" in joined                                # the gate clone is caught
    assert "beat_index" not in joined                             # a real attribute is never flagged


def test_freezelint_phantom_attr_spy_noop_until_built(tmp_path):
    from raster.freezelint import lint_phantom_attr_spies
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)                            # package dir exists but EMPTY (no source)
    (code / "tests").mkdir()
    (code / "tests" / "test_x.py").write_text(
        "def test_y(obj):\n    assert getattr(obj, '_step_count', 0) == 1\n")
    assert lint_phantom_attr_spies(code, "pkg") == []             # no product tokens yet -> no-op mid-freeze


def _clean_review_project(tmp_path):
    # a spec whose lint is clean: P0 owns the frozen tests, M0.T1 delivers only product code.
    spec = {"meta": {"package": "pkg", "project": "P"}, "modules": [
        {"id": "P0", "tasks": [{"id": "P0.M0", "deliverables": ["tests/test_smoke.py", "tests/gate.py"],
                                "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest --collect-only -q tests/"}}]},
        {"id": "M0", "tasks": [{"id": "M0.T1", "deliverables": ["pkg/__init__.py"],
                                "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest -q tests/test_smoke.py"}}],
         "gate": {"id": "G0", "integration_test": {"file": "tests/gate.py", "cmd": "pytest -q tests/gate.py"}}}]}
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / "tests").mkdir()
    return Project(root=tmp_path, code=code, cfg=Config(), ry={"project": "P", "package": "pkg"}, spec=spec)


def test_freeze_review_flags_green_at_head(tmp_path, monkeypatch, capsys):
    # The freeze-review gate EXECUTES red-before-green: a task test or gate that PASSES at HEAD
    # (deliverable absent) is blind/false-green and must BLOCK queue; a test that fails is fine.
    from types import SimpleNamespace
    from raster import freeze_review
    project = _clean_review_project(tmp_path)
    (project.code / "tests" / "test_smoke.py").write_text("def test_blind():\n    assert 1 == 1\n")  # green at HEAD
    (project.code / "tests" / "gate.py").write_text("def test_gate():\n    assert 1 == 1\n")          # green gate
    monkeypatch.setattr(freeze_review, "load_project", lambda d: project)
    # simulate running each frozen test against the real tree: both are GREEN at HEAD (the defect)
    monkeypatch.setattr(freeze_review.execlib, "run_test", lambda proj, cmd, stub_pkg=None: (True, "1 passed"))
    rc = freeze_review.run_freeze_review(SimpleNamespace(dir=str(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 1                                                # green-at-HEAD blocks
    assert "GREEN at HEAD" in out
    assert "M0.T1" in out and "G0" in out                        # both the task test AND the gate flagged
    assert "BLOCKING" in out


def test_freeze_review_passes_when_red(tmp_path, monkeypatch, capsys):
    # All frozen tests fail at HEAD (product absent) -> red-before-green holds -> mechanical checks pass.
    from types import SimpleNamespace
    from raster import freeze_review
    project = _clean_review_project(tmp_path)
    (project.code / "tests" / "test_smoke.py").write_text("import pkg\ndef test_x():\n    assert pkg\n")
    (project.code / "tests" / "gate.py").write_text("import pkg\ndef test_g():\n    assert pkg\n")
    monkeypatch.setattr(freeze_review, "load_project", lambda d: project)
    monkeypatch.setattr(freeze_review.execlib, "run_test",
                        lambda proj, cmd, stub_pkg=None: (False, "1 failed (ModuleNotFoundError)"))
    rc = freeze_review.run_freeze_review(SimpleNamespace(dir=str(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 0
    assert "correctly RED at HEAD" in out
    assert "mechanical checks clean" in out


def test_freezelint_half_matrix_lookup(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # DISTANCES is a half-matrix: one order per pair, no diagonal. Looked up by free (a,b) it is a
    # non-reflexive, asymmetric pseudo-metric -> any value computed under a real metric is unsatisfiable.
    (tests / "test_seg.py").write_text(
        'DISTANCES = {("C", "Em"): 0.5, ("C", "G"): 0.8}\n'
        "def metric(a, b):\n    return DISTANCES.get((a, b), 1.0)\n"     # free-var lookup -> flagged
        'def test_known():\n    assert DISTANCES[("C", "Em")] == 0.5\n')  # constant key -> NOT flagged
    viols = [x for x in lint_frozen_tests(code, "pkg") if "HALF-MATRIX" in x]
    assert len(viols) == 1 and "DISTANCES" in viols[0]    # ONLY the free-var (a,b) .get is flagged
    assert ":3:" in viols[0]                               # the constant-key subscript was not flagged


def test_freezelint_full_symmetric_table_clean(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # A FULL symmetric table with diagonal IS a total metric -> a free-var lookup is safe, not flagged.
    (tests / "test_seg.py").write_text(
        'D = {("C","C"): 0.0, ("Em","Em"): 0.0, ("C","Em"): 0.5, ("Em","C"): 0.5}\n'
        "def metric(a, b):\n    return D.get((a, b), 1.0)\n"
        "def test_x():\n    assert metric('C', 'C') == 0.0\n")
    assert lint_frozen_tests(code, "pkg") == []           # reflexive + symmetric -> no false positive


def test_freezelint_stochastic_per_step_gate(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # The G5 fingerprint: per-step monotonicity / dip-count threshold on np.diff of a trend.
    (tests / "gate_obs.py").write_text(
        "import numpy as np\n"
        "AVG_TREND = np.array([0.1, 0.2, 0.15, 0.3])\n"
        "def test_dip_count():\n"
        "    diffs = np.diff(AVG_TREND)\n"
        "    assert np.sum(diffs < -1e-6) <= 1\n"            # dip-count vs constant -> flagged
        "def test_monotone():\n"
        "    assert np.all(np.diff(AVG_TREND) >= 0)\n")      # per-step all() monotone -> flagged
    viols = lint_frozen_tests(code, "pkg")
    assert any("dip-count threshold" in v for v in viols)
    assert any("per-step monotonicity" in v for v in viols)
    assert len(viols) == 2                                   # exactly the two smells, nothing else


def test_freezelint_distributional_gate_clean(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # The RECOMMENDED encoding: net rise + majority-up (count-vs-COUNT, not count-vs-constant).
    # Neither np.sum(diffs) (a net total, no sign-compare) nor a count-vs-count is the smell.
    (tests / "gate_obs.py").write_text(
        "import numpy as np\n"
        "AVG_TREND = np.array([0.1, 0.2, 0.15, 0.31])\n"
        "def test_net_rise():\n"
        "    assert AVG_TREND[-1] - AVG_TREND[0] >= 0.05\n"   # net rise, no np.diff -> clean
        "def test_majority_up():\n"
        "    diffs = np.diff(AVG_TREND)\n"
        "    assert np.sum(diffs > 1e-6) > np.sum(diffs < -1e-6)\n"  # count-vs-count -> clean
        "    assert np.sum(diffs) >= 0.05\n")                 # sum of diffs (net), not a sign-count -> clean
    assert lint_frozen_tests(code, "pkg") == []


def test_freezelint_constant_parameter_conflation(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # DIATONIC_CHORDS is the structural vocabulary (7); n_chord_types=3 is a run knob. Asserting
    # len(VOCAB) == 3 pins a constant's size to the knob's value -> unsatisfiable layer confusion.
    (tests / "test_integrity.py").write_text(
        'DIATONIC_CHORDS = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]\n'
        "def cfg():\n    return dict(n_chord_types=3)\n"
        "def test_integrity():\n    assert len(DIATONIC_CHORDS) == 3\n")        # 3 is the knob -> flagged
    viols = [x for x in lint_frozen_tests(code, "pkg") if "LAYER CONFUSION" in x]
    assert len(viols) == 1
    assert "DIATONIC_CHORDS" in viols[0] and "n_chord_types" in viols[0]


def test_freezelint_constant_parameter_conflation_clean(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # Two clean forms: the DERIVED expected (no int literal), and a literal that is NOT a knob value.
    (tests / "test_integrity.py").write_text(
        'DIATONIC_CHORDS = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]\n'
        "def diatonic_major():\n    return DIATONIC_CHORDS\n"
        "def cfg():\n    return dict(n_chord_types=3)\n"
        "def test_derived():\n    assert len(DIATONIC_CHORDS) == len(diatonic_major())\n"   # derived
        "def test_structural():\n    assert len(DIATONIC_CHORDS) == 7\n")                   # 7 != any knob
    assert [x for x in lint_frozen_tests(code, "pkg") if "LAYER CONFUSION" in x] == []


def test_failure_signature():
    out = ("tests/test_seg.py::test_seg FAILED\n"
           "    def test_seg():\n"
           ">       assert segregation_index(g, metric) == 0.8\n"
           "E       assert 0.0833 == 0.8\n"
           "FAILED tests/test_seg.py::test_seg - assert 0.0833 == 0.8\n"
           "=== 1 failed in 0.3s ===\n")
    sig = execlib.failure_signature(out)
    assert "0.0833 == 0.8" in sig                          # carries the concrete value
    assert "0.3s" not in sig and "def test_seg" not in sig # drops volatile timing / source context
    # byte-stable across runs that differ only in timing -> a repeat IS an oracle-bug plateau signal
    assert sig == execlib.failure_signature(out.replace("0.3s", "1.7s"))
    assert execlib.failure_signature("=== 5 passed in 0.1s ===") == ""


def test_lint_copied_constants_flags_based_on_copy(tmp_path):
    from raster.freezelint import lint_copied_constants
    code = tmp_path / "code"
    pkg = code / "pkg"
    pkg.mkdir(parents=True)
    # canonical source: chords.py owns the vocabulary; consumers must import it.
    (pkg / "chords.py").write_text("DIATONIC = {'C': {0, 4, 7}, 'Em': {4, 7, 11}}\n")
    # sonify.py holds a PRIVATE copy 'based on chords.py' -> the M6 tautology -> flagged.
    (pkg / "sonify.py").write_text(
        "# Based on the diatonic vocabulary in chords.py\n"
        "PALETTE = {'C': [0, 4, 7], 'Em': [4, 7, 11]}\n"
        "def render(name):\n    return PALETTE[name]\n")
    v = "\n".join(lint_copied_constants(code, "pkg"))
    assert "sonify.py" in v and "PALETTE" in v and "chords" in v
    assert "transcription agreement" in v.lower() or "TRANSCRIPTION" in v


def test_lint_copied_constants_flags_peer_copies(tmp_path):
    from raster.freezelint import lint_copied_constants
    code = tmp_path / "code"
    pkg = code / "pkg"
    pkg.mkdir(parents=True)
    # same constant NAME defined as a literal in two product modules -> peer copies.
    (pkg / "model.py").write_text("WEIGHTS = [1, 2, 3]\n")
    (pkg / "report.py").write_text("WEIGHTS = [1, 2, 3]\n")
    v = "\n".join(lint_copied_constants(code, "pkg"))
    assert "WEIGHTS" in v and "model.py" in v and "report.py" in v and "peer copies" in v


def test_lint_copied_constants_clean_when_derived(tmp_path):
    from raster.freezelint import lint_copied_constants
    code = tmp_path / "code"
    pkg = code / "pkg"
    pkg.mkdir(parents=True)
    # one canonical literal; the consumer IMPORTS it (no private copy) -> clean.
    (pkg / "chords.py").write_text("DIATONIC = {'C': {0, 4, 7}}\n")
    (pkg / "sonify.py").write_text(
        "from pkg.chords import DIATONIC\n"
        "def render(name):\n    return DIATONIC[name]\n")
    (pkg / "__init__.py").write_text("__all__ = ['render']\n")   # dunder, repeated -> never flagged
    (pkg / "cli.py").write_text("__all__ = ['main']\n")
    assert lint_copied_constants(code, "pkg") == []


def test_lint_copied_constants_noop_when_unbuilt(tmp_path):
    from raster.freezelint import lint_copied_constants
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)                        # package dir exists, no .py files yet
    assert lint_copied_constants(code, "pkg") == []


def test_lint_dead_modules_flags_islands(tmp_path):
    from raster.freezelint import lint_dead_modules
    code = tmp_path / "code"
    pkg = code / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from pkg.model import Model\n")  # public API re-export
    (pkg / "model.py").write_text("from pkg.metrics import jaccard\n")  # imports metrics
    (pkg / "metrics.py").write_text("def jaccard(a, b):\n    return 0.0\n")  # consumed -> live
    (pkg / "policy.py").write_text("def relocate():\n    pass\n")      # imported by NOBODY -> island
    (pkg / "cli.py").write_text("if __name__ == '__main__':\n    print('go')\n")  # entrypoint -> ok
    spec = {"meta": {"package": "pkg"}, "modules": [{"id": "M1", "tasks": [
        {"id": "M1.T1", "deliverables": ["pkg/__init__.py", "pkg/model.py", "pkg/metrics.py",
                                         "pkg/policy.py", "pkg/cli.py"]}]}]}
    v = "\n".join(lint_dead_modules(code, "pkg", spec))
    assert "pkg.policy" in v and "island" in v          # delivered, exists, imported by nothing
    assert "pkg.metrics" not in v                        # imported by model -> not flagged
    assert "pkg.model" not in v                          # re-exported by the package root __init__
    assert "pkg.cli" not in v                             # __main__ entrypoint excluded
    assert "pkg:" not in v and "pkg " not in v           # package root never flagged


def test_lint_dead_modules_noop_when_unbuilt(tmp_path):
    from raster.freezelint import lint_dead_modules
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)                    # declared but no module files on disk
    spec = {"meta": {"package": "pkg"}, "modules": [{"id": "M1", "tasks": [
        {"id": "M1.T1", "deliverables": ["pkg/model.py", "pkg/metrics.py"]}]}]}
    assert lint_dead_modules(code, "pkg", spec) == []     # self-limits to existing modules


def test_lint_dead_modules_relative_import(tmp_path):
    from raster.freezelint import lint_dead_modules
    code = tmp_path / "code"
    pkg = code / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .model import Model\n")    # re-export keeps model live
    (pkg / "model.py").write_text("from . import metrics\n")          # relative sibling import
    (pkg / "metrics.py").write_text("def jaccard(a, b):\n    return 0.0\n")
    spec = {"meta": {"package": "pkg"}, "modules": [{"id": "M1", "tasks": [
        {"id": "M1.T1", "deliverables": ["pkg/__init__.py", "pkg/model.py", "pkg/metrics.py"]}]}]}
    assert lint_dead_modules(code, "pkg", spec) == []     # relative import marks metrics reachable


def _blind_spec(deliverables):
    # one IMPLEMENT module/task whose frozen test is tests/test_demo.py
    return {"meta": {"package": "pkg"}, "modules": [
        {"id": "M8", "tasks": [
            {"id": "M8.T1", "deliverables": deliverables,
             "unit_test": {"file": "tests/test_demo.py", "cmd": "pytest -q"}}]}]}


def test_lint_deliverable_blind_tests_flags_missing_reference(tmp_path):
    from raster.freezelint import lint_deliverable_blind_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # the M8.T1 fingerprint: the test checks an inline kwargs dict + golden constants and never
    # references configs/demo.yaml -> green at HEAD with the artifact absent.
    (tests / "test_demo.py").write_text(
        "from pkg.config import Config\n"
        "GOLDEN = {'a': 1}\n"
        "def test_inline():\n    assert Config(n_chord_types=3).n_chord_types == 3\n")
    v = "\n".join(lint_deliverable_blind_tests(code, "pkg", _blind_spec(["configs/demo.yaml"])))
    assert "M8.T1" in v and "demo.yaml" in v               # names the task and the blind deliverable
    assert "red-before-green" in v.lower()                  # the decisive contract


def test_lint_deliverable_blind_tests_clean_when_referenced(tmp_path):
    from raster.freezelint import lint_deliverable_blind_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # the test loads the deliverable by path through the real loader -> references it -> clean.
    (tests / "test_demo.py").write_text(
        "from pathlib import Path\n"
        "from pkg.config import load\n"
        "def test_loads():\n"
        "    cfg = load(Path(__file__).parent.parent / 'configs' / 'demo.yaml')\n"
        "    assert cfg.n_chord_types >= 2\n")
    assert lint_deliverable_blind_tests(code, "pkg", _blind_spec(["configs/demo.yaml"])) == []


def test_lint_deliverable_blind_tests_ignores_py_and_packaging(tmp_path):
    from raster.freezelint import lint_deliverable_blind_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    # a test that references NONE of the deliverables, but they're all out of scope: a .py module
    # (import-referenced, covered elsewhere), pyproject.toml (packaging, never test-loaded), and a
    # directory deliverable (no single file to grep).
    (tests / "test_demo.py").write_text("def test_x():\n    assert True\n")
    spec = _blind_spec(["pkg/foo.py", "pyproject.toml", "assets/"])
    assert lint_deliverable_blind_tests(code, "pkg", spec) == []


def test_lint_deliverable_blind_tests_skips_authoring_and_unbuilt(tmp_path):
    from raster.freezelint import lint_deliverable_blind_tests
    code = tmp_path / "code"
    tests = code / "tests"
    tests.mkdir(parents=True)
    (tests / "test_demo.py").write_text("def test_x():\n    assert True\n")
    # a P0 AUTHORING task owns its tests; never treated as a deliverable-blind impl task
    p0 = {"meta": {"package": "pkg"}, "modules": [
        {"id": "P0", "tasks": [
            {"id": "P0.M8", "deliverables": ["configs/demo.yaml"],
             "unit_test": {"file": "tests/test_demo.py", "cmd": "pytest -q"}}]}]}
    assert lint_deliverable_blind_tests(code, "pkg", p0) == []
    # and an impl task whose frozen test isn't authored yet -> no-op (self-limits to files on disk)
    spec = _blind_spec(["configs/missing.yaml"])
    spec["modules"][0]["tasks"][0]["unit_test"]["file"] = "tests/not_written.py"
    assert lint_deliverable_blind_tests(code, "pkg", spec) == []


def test_freezelint_clean_suite(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    (code / "tests").mkdir(parents=True)
    (code / "tests" / "test_ok.py").write_text(
        'D = {"a": 1}\ndef test_x():\n    assert D["a"] == 1\n')
    assert lint_frozen_tests(code, "pkg") == []                # no tests/ defects, no false positives


def test_cli_lint_clean(tmp_path, monkeypatch, capsys):
    # a spec with NO impl-task tests/ deliverable (the shared SPEC has one, by design, to test
    # prompt filtering — that would now trip lint_spec), and no tests/ dir -> fully clean.
    clean = {"meta": {"package": "pkg", "project": "P"}, "modules": [
        {"id": "M0", "tasks": [{"id": "M0.T1", "deliverables": ["pkg/__init__.py"]}]}]}
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    code = tmp_path / "code"
    (code / "designdocs").mkdir(parents=True)
    (code / "pkg").mkdir()
    (code / "designdocs" / "tasks.yaml").write_text(yaml.safe_dump(clean))
    (code / "raster.yaml").write_text(yaml.safe_dump({"project": "P", "package": "pkg"}))
    assert main(["lint", "--dir", str(tmp_path)]) == 0
    assert "clean" in capsys.readouterr().out


def test_lint_spec_flags_impl_task_test_deliverable():
    from raster.spec import lint_spec
    # M0.T1 (an IMPLEMENT task) lists tests/test_smoke.py as a deliverable — unsatisfiable.
    v = "\n".join(lint_spec(SPEC))
    assert "M0.T1" in v and "tests/test_smoke.py" in v
    # P0 authoring tasks legitimately own tests/ paths -> never flagged.
    assert "P0.M0" not in v


def test_write_files_strips_double_root_prefix(tmp_path):
    project = make_project(tmp_path)                           # root name is "code"
    # the worker re-prefixes the root it was told paths are "relative to" -> code/pkg/x.py
    written = execlib.write_files(project, {"code/pkg/x.py": "x = 1",
                                            "code/pyproject.toml": "[build]"}, allow_tests=False)
    assert written == ["pkg/x.py", "pyproject.toml"]           # stripped back to the real rel
    assert (project.code / "pkg" / "x.py").read_text() == "x = 1\n"
    assert not (project.code / "code").exists()                # never landed at code/code/
    assert (project.code / "pyproject.toml").exists()


def test_output_contract_prohibits_root_prefix():
    c = execlib.output_contract("pkg", "code")
    assert "do NOT prefix paths with `code/`" in c
    assert "not `code/pkg/example.py`" in c                    # negative example present


def test_missing_product_import_detection():
    from raster.build import _missing_product_import
    assert _missing_product_import("ModuleNotFoundError: No module named 'postineq'", "postineq")
    assert _missing_product_import("No module named 'postineq.config'", "postineq")
    assert not _missing_product_import("No module named 'numpy'", "postineq")
    assert not _missing_product_import("anything", "")


def test_normalize_pytest_cmd():
    out = execlib.normalize_pytest_cmd("pytest -q tests/")
    assert out == f"{sys.executable} -m pytest -q tests/"
    # idempotent: don't double-wrap
    assert execlib.normalize_pytest_cmd(out) == out


def test_summarize_pytest():
    s = execlib.summarize_pytest("...\n=== 2 failed, 45 passed in 1.2s ===\n")
    assert "2 failed, 45 passed" in s


def test_build_prompt_impl_filters_tests_and_names_package(tmp_path):
    project = make_project(tmp_path)
    (project.code / "tests").mkdir()
    (project.code / "tests" / "test_smoke.py").write_text("def test_x(): assert pkg")
    module, task = find_task(SPEC, "M0.T1")
    prompt = execlib.build_prompt(project, module, task, authoring=False)
    assert "building the P project" in prompt
    assert "pkg/example.py" in prompt                   # output contract uses the package
    assert "FROZEN unit test" in prompt and "def test_x" in prompt
    assert "tests/test_smoke.py" not in str(             # tests/ deliverable hidden from impl
        prompt.split("Deliverables")[1].splitlines()[0])


def test_build_prompt_authoring_lists_behaviors(tmp_path):
    project = make_project(tmp_path)
    module, task = find_task(SPEC, "P0.M0")
    prompt = execlib.build_prompt(project, module, task, authoring=True)
    assert "AUTHORING FROZEN TESTS" in prompt
    assert "Behaviors to cover (module M0)" in prompt and "M0.T1" in prompt
    assert "Module gate to author (G0)" in prompt


# ------------------------------------------------------------- escalation ladder
def test_ladder_resolves_default_and_custom(tmp_path):
    project = make_project(tmp_path)
    assert project.ladder() == DEFAULT_LADDER                 # SPEC has no meta.ladder
    project.spec = {**SPEC, "meta": {**SPEC["meta"],
                    "ladder": [{"worker": "worker"}, {"worker": "strong", "think": True}]}}
    assert project.ladder() == [{"worker": "worker", "think": False},
                                {"worker": "strong", "think": True}]


def test_start_index_is_the_floor():
    ladder = DEFAULT_LADDER
    assert start_index(ladder, "worker") == 0                 # cheap task starts at the bottom
    assert start_index(ladder, "strong") == 1                 # strong task floors above `worker`
    assert start_index(ladder, "mystery") == 0                # unknown -> bottom


def test_rung_index_dwells_then_climbs():
    # escalate_after=2, 3-rung ladder, starting at rung 0: dwell 2 attempts, then climb one/attempt
    seq = [rung_index(0, a, 2, 3) for a in range(1, 6)]
    assert seq == [0, 0, 1, 2, 2]                             # capped at the top rung
    # a strong-start task (start=1) reaches the think-on rung on the first escalation
    assert [rung_index(1, a, 2, 3) for a in range(1, 5)] == [1, 1, 2, 2]


def test_resolve_rung_think_precedence():
    ladder = DEFAULT_LADDER
    # implementation, no pins -> rung's own think (off at the bottom, on at the top)
    assert resolve_rung(ladder, 0, {"worker": "worker"}, False, {}) == ("worker", False)
    assert resolve_rung(ladder, 2, {"worker": "strong"}, False, {}) == ("strong", True)
    # P0 authoring is ALWAYS think-on, even on a think-off rung
    assert resolve_rung(ladder, 1, {"worker": "strong"}, True, {}) == ("strong", True)
    # a per-task think pin wins over both the rung and the authoring default
    assert resolve_rung(ladder, 2, {"worker": "strong", "think": False}, True, {}) == ("strong", False)
    # a global meta.think pin overrides the rung for implementation tasks
    assert resolve_rung(ladder, 2, {"worker": "strong"}, False, {"think": False}) == ("strong", False)


# ------------------------------------------------------------------- queue / find
def test_linearize_chain(tmp_path):
    project = make_project(tmp_path)
    chain = linearize(project, exec_cmd="raster")
    # the freeze-review gate is inserted on the P0->impl boundary (after authoring, before M0)
    assert [c["id"] for c in chain] == ["P0.M0", "freeze-review", "M0.T1", "G0"]
    assert chain[0]["command"] == "raster build P0.M0" and chain[0]["resource"] == 2
    fr = chain[1]
    assert fr["command"] == "raster freeze-review" and fr["resource"] == 3   # gate -> cpu, fails closed
    assert chain[3]["command"] == "raster test G0" and chain[3]["resource"] == 3  # gate -> cpu
    # titles use the `raster:` prefix (not the project name) — the trundlr project groups them
    assert [c["title"] for c in chain] == ["raster: P0.M0", "raster: freeze-review",
                                           "raster: M0.T1", "raster: G0"]


def test_linearize_budget_sets_duration(tmp_path):
    # a per-task/gate `budget:` (seconds) reserves at least budget/3600 hours in the trundlr chain,
    # so a legitimately long GA gate gets a scheduling window matching its timeout.
    project = make_project(tmp_path)
    project.spec = {"meta": SPEC["meta"], "execution": SPEC["execution"], "modules": [
        {"id": "M7", "name": "ga", "tasks": [{"id": "M7.T1", "title": "GA", "worker": "strong",
                                              "budget": 7200}],
         "gate": {"id": "G7", "spec": "undo", "budget": 5400,
                  "integration_test": {"file": "tests/gate_undo.py", "cmd": "pytest -q"}}}]}
    chain = {c["id"]: c for c in linearize(project, exec_cmd="raster")}
    assert chain["M7.T1"]["duration"] == 2.0          # 7200s = 2h overrides the 1.75h think-on prior
    assert chain["G7"]["duration"] == 1.5             # 5400s = 1.5h overrides the 0.1h gate prior
    # no P0 authoring module here -> no freeze-review node inserted
    assert "freeze-review" not in chain


def test_linearize_inserts_review_checkpoints(tmp_path):
    project = make_project(tmp_path)
    project.cfg.human_resource = 1
    project.cfg.claude_resource = 4
    project.spec = {
        "meta": SPEC["meta"], "execution": SPEC["execution"],
        "modules": [
            {"id": "M0", "name": "scaffold",
             "checkpoint": "Claude — review the frozen suite; go/no-go.",
             "tasks": [{"id": "M0.T1", "title": "scaffold", "worker": "worker"}],
             "gate": {"id": "G0", "spec": "imports",
                      "integration_test": {"file": "tests/g.py", "cmd": "pytest -q"}}},
        ],
        "final_checkpoint": "Claude — whole-system sign-off.",
    }
    chain = linearize(project, exec_cmd="raster")
    # a checkpoint is inserted on the edge BEFORE its module, and one final sign-off at the end
    assert [c["id"] for c in chain] == ["M0", "M0.T1", "G0", "final"]
    ck = chain[0]
    assert ck["kind"] == "checkpoint" and ck["command"] is None      # null command => blocks
    assert ck["resources"] == [1, 4]                                  # [human, claude]
    assert ck["title"] == "raster: checkpoint M0"
    assert ck["description"] == "Claude — review the frozen suite; go/no-go."
    fin = chain[-1]
    assert fin["command"] is None and fin["resources"] == [1, 4]
    # with reviewer resources unset, the checkpoint queues reviewer-less (still blocks)
    project.cfg.human_resource = project.cfg.claude_resource = 0
    assert linearize(project, exec_cmd="raster")[0]["resources"] == []


def test_estimate_hours_think_and_history():
    from raster.queue import estimate_hours
    # gates and checkpoints are flat (and a checkpoint must stay > 0 for trundlr)
    assert estimate_hours("gate", "") == 0.1
    assert estimate_hours("checkpoint", "") == 1.0
    # think flips a strong task from the cheap prior to the measured think-on cost
    assert estimate_hours("task", "strong", think=False) == 0.5
    assert estimate_hours("task", "strong", think=True) == 1.75
    # a think-OFF strong floor still reserves at think-on — it is one climb away
    assert estimate_hours("task", "strong", escalates=True) == 1.75
    # a worker-floored task stays cheap; it is not reserved for a climb
    assert estimate_hours("task", "worker") == 0.25
    # recent SIMILAR runs override the prior; no history yet -> fall back to the prior
    hist = [{"worker": "strong", "think": True, "hours": 1.0},
            {"worker": "strong", "think": True, "hours": 3.0},
            {"worker": "worker", "think": True, "hours": 9.0}]   # different tier, ignored
    assert estimate_hours("task", "strong", think=True, history=hist) == 3.0  # median of [1,3]
    assert estimate_hours("task", "strong", think=True, history=[]) == 1.75


def test_linearize_budgets_think_on_tasks(tmp_path):
    project = make_project(tmp_path)
    chain = {c["id"]: c for c in linearize(project, exec_cmd="raster")}
    assert chain["P0.M0"]["duration"] == 1.75   # P0 authoring is always think-on
    assert chain["M0.T1"]["duration"] == 0.25   # worker-floored impl stays cheap
    assert chain["G0"]["duration"] == 0.1


def test_find_gate_and_task():
    assert find_task(SPEC, "M0.T1")[1]["title"] == "Package scaffold"
    assert find_gate(SPEC, "G0")[1]["id"] == "G0"
    assert find_task(SPEC, "nope") == (None, None)


# ---------------------------------------------------- trundlr project name -> id
def test_resolve_project_id_finds_then_creates(monkeypatch):
    from raster import trundlr
    posted = []

    def fake_api(api, method, path, body=None, timeout=30):
        if method == "GET" and path == "/projects/":
            return [{"id": 7, "name": "existing"}, {"id": 9, "name": "other"}]
        if method == "POST" and path == "/projects/":
            posted.append(body)
            return {"id": 42, "name": body["name"]}
        raise AssertionError((method, path))

    monkeypatch.setattr(trundlr, "_api", fake_api)
    assert trundlr.resolve_project_id("http://x", "existing") == (7, False)   # matched, no create
    assert trundlr.resolve_project_id("http://x", "brandnew", folder="/r") == (42, True)
    assert posted == [{"name": "brandnew", "priority": 1, "folder": "/r"}]
    assert trundlr.resolve_project_id("http://x", "missing", create=False) == (None, False)


def test_cache_project_id_rewrites_value_keeps_comment(tmp_path):
    from raster.queue import _cache_project_id
    project = make_project(tmp_path)
    ry = project.code / "raster.yaml"
    ry.write_text("trundlr:\n  project_id: my demo   # set by init\n  api_url: http://x\n")
    _cache_project_id(project, 42)
    out = ry.read_text()
    assert "  project_id: 42   # set by init\n" in out   # value replaced, comment + spacing kept
    assert "api_url: http://x" in out                    # other lines untouched


# ------------------------------------------------------------------- CLI dry-runs
def _on_disk_project(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    code = tmp_path / "code"
    (code / "designdocs").mkdir(parents=True)
    (code / "pkg").mkdir()
    (code / "designdocs" / "tasks.yaml").write_text(yaml.safe_dump(SPEC))
    (code / "raster.yaml").write_text(yaml.safe_dump({"project": "P", "package": "pkg"}))
    return tmp_path


def test_cli_build_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["build", "M0.T1", "--dir", str(root), "--dry-run"]) == 0
    assert "building the P project" in capsys.readouterr().out


def test_cli_test_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["test", "G0", "--dir", str(root), "--dry-run"]) == 0
    assert "gate G0" in capsys.readouterr().out


def test_cli_queue_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["queue", "--dir", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "raster build P0.M0" in out and "raster test G0" in out
