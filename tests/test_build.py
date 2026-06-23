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


def test_freezelint_clean_suite(tmp_path):
    from raster.freezelint import lint_frozen_tests
    code = tmp_path / "code"
    (code / "tests").mkdir(parents=True)
    (code / "tests" / "test_ok.py").write_text(
        'D = {"a": 1}\ndef test_x():\n    assert D["a"] == 1\n')
    assert lint_frozen_tests(code, "pkg") == []                # no tests/ defects, no false positives


def test_cli_lint_clean(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)             # has code/ but no tests/
    assert main(["lint", "--dir", str(root)]) == 0
    assert "clean" in capsys.readouterr().out


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
    assert [c["id"] for c in chain] == ["P0.M0", "M0.T1", "G0"]
    assert chain[0]["command"] == "raster build P0.M0" and chain[0]["resource"] == 2
    assert chain[2]["command"] == "raster test G0" and chain[2]["resource"] == 3  # gate -> cpu
    # titles use the `raster:` prefix (not the project name) — the trundlr project groups them
    assert [c["title"] for c in chain] == ["raster: P0.M0", "raster: M0.T1", "raster: G0"]


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
