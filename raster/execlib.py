"""Shared executor guts for `raster build` (coding) and `raster test` (assessment):
prompt building, FILE-block parse/write, pytest running, and commit+push on pass.

Generalized from the SchellingChords doer — the package name, project name, and
project description are read from the Project (raster.yaml / tasks.yaml meta), not
hardcoded. Commits use the machine config's non-PII identity with NO co-authorship.
"""

import ast
import builtins
import os
import re
import subprocess
import sys
from pathlib import Path

from raster.runlog import log
from raster.spec import Project, module_by_id, owner_of

FILE_RE = re.compile(r"=== FILE: (.+?) ===\n(.*?)\n=== END FILE ===", re.DOTALL)
# Opening marker alone (no `=== END FILE ===` required). An opened path absent from FILE_RE's
# parse is a block missing its terminator — single-line match, so a stray `===` in a file body
# (DOTALL off here) can't be swallowed into it.
OPEN_RE = re.compile(r"=== FILE: (.+?) ===")
TEST_TIMEOUT = int(os.environ.get("RASTER_TEST_TIMEOUT", 600))
GIT_PUSH = os.environ.get("RASTER_GIT_PUSH", "1") not in ("0", "false", "no", "")


def output_contract(pkg: str, root: str = "code") -> str:
    example = f"{pkg}/example.py" if pkg else "example.py"
    bad = f"{root}/{example}"
    toml_bad = f"{root}/pyproject.toml"
    return ("Output ONLY files, each wrapped EXACTLY like this (no prose, no markdown fences):\n\n"
            f"=== FILE: {example} ===\n"
            "<full file contents>\n"
            "=== END FILE ===\n\n"
            "Emit every file in full (not a diff). Emit each path RELATIVE TO the "
            f"`{root}/` root — do NOT prefix paths with `{root}/`. "
            f"Write `{example}`, not `{bad}`; write `pyproject.toml`, not `{toml_bad}`.")


def parse_files(text: str) -> dict:
    return {path.strip(): content for path, content in FILE_RE.findall(text)}


def parse_diagnostics(text: str, files: dict) -> dict:
    """Classify a FILE-block parse for self-diagnosing logs and targeted re-prompts.
    `opened` = every `=== FILE: ===` opening marker; `parsed` = the blocks FILE_RE actually
    closed; `unterminated` = opened-but-not-closed paths (a block missing `=== END FILE ===`).
    A non-empty `unterminated` ALONGSIDE a non-empty `files` is the partial-parse silent-drop
    case — some files landed, one was discarded with no re-prompt (write-path addendum, W)."""
    opened = [p.strip() for p in OPEN_RE.findall(text)]
    parsed = set(files)
    return {"opened": opened, "parsed": list(files),
            "unterminated": [p for p in opened if p not in parsed]}


def parse_failure_reason(diag: dict) -> str:
    """Why a parse produced (too) few files, so a human reading the doer log classifies it at a
    glance instead of re-deriving it from the reply head (write-path addendum, V)."""
    if diag["unterminated"]:
        return (f"opened {len(diag['unterminated'])} FILE block(s) but no `=== END FILE ===` "
                f"closer: {diag['unterminated']}")
    return "no `=== FILE:` opening marker at all"


def reprompt_for_parse_failure(diag: dict, root: str = "code") -> str:
    """A TARGETED re-prompt naming the exact defect and exact remedy — never a restated spec.
    A generic 're-emit using the format' re-sends the contract the worker already had in view
    and frequently reproduces the identical drift, burning a second ~cycle; naming the
    unterminated path(s) and the precise closer converts most of these to one-cycle recoveries
    (write-path addendum, U — the cheapest high-leverage write-path fix there is)."""
    bad = diag["unterminated"]
    if bad:
        paths = ", ".join(bad)
        return (f"You opened {len(bad)} FILE block(s) ({paths}) but did NOT close them. Each "
                "file must end with a line containing EXACTLY `=== END FILE ===` — not ``` , not "
                "`=== END ===`, not `=== FILE END ===`. Re-emit ALL files in full, each wrapped "
                "`=== FILE: <path> ===` then the full contents then `=== END FILE ===`.")
    return ("No `=== FILE: <path> ===` marker was found in your reply. Output ONLY files, each "
            "wrapped EXACTLY as `=== FILE: <path> ===`, then the full contents, then a line "
            f"`=== END FILE ===` — no prose, no markdown fences. Paths relative to the {root}/ "
            f"root, not prefixed with {root}/.")


# Names that are always resolvable without an explicit binding: every builtin, plus the module
# dunders the interpreter injects. Used by the static undefined-name pass below.
_ALWAYS_BOUND = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__loader__",
    "__package__", "__path__", "__dict__", "__class__", "__module__", "__qualname__",
    "__annotations__", "__all__",
}


def _bound_names(tree: ast.AST) -> set:
    """Every name BOUND anywhere in `tree`, scope-INSENSITIVELY (a flat over-approximation):
    Store-context targets (assign / for / with-as / comprehension / walrus), function & class
    names, every parameter, import aliases, except-handler names, and global/nonlocal declarations.
    Scope-insensitive ON PURPOSE — a name bound in ANY scope is treated as defined — so this can
    only ever UNDER-report an undefined name, never invent one. Zero false positives is the whole
    point: a finding here must be a name that is genuinely unbound module-wide."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def _undefined_in_tree(tree: ast.AST) -> list:
    """Conservative undefined-name findings in one module tree: Load-context names bound in NO
    scope and not builtin, returned as [(lineno, name)] (first use of each, in order). BAILS
    (returns []) when the file `import *`s or uses a `match` statement — there we can't see every
    binding, so a finding could be wrong and we'd rather miss than misfire."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return []
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            return []
    bound = _bound_names(tree) | _ALWAYS_BOUND
    seen, out = set(), []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id not in bound and node.id not in seen):
            seen.add(node.id)
            out.append((node.lineno, node.id))
    return out


def undefined_names(project: Project) -> list:
    """Scan the product package for names USED but defined in no scope and not builtin — the
    masking-runtime-error class (a missing `import Path`, a never-defined helper `jaccard_distance`).
    The interpreter raises only the FIRST such NameError per run, so a stack of them peels one per
    ~test-cycle (~40 min each); surfacing the WHOLE list at once collapses that chain into a single
    repair turn (changing-failure-chain guidance, FF). Conservative & scope-insensitive (see
    `_bound_names`): it only flags names bound NOWHERE in their file, so it under-reports rather than
    ever inventing a finding — it feeds the worker extra signal, it never red-lights a build on its
    own. Returns [(relpath, lineno, name)]; [] when clean or the package isn't built yet."""
    pkg = project.code / project.package if project.package else project.code
    if not pkg.is_dir():
        return []
    out = []
    for f in sorted(pkg.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except (SyntaxError, OSError):
            continue                       # a syntax error is a louder failure the test already shows
        rel = f.relative_to(project.code).as_posix()
        for lineno, name in _undefined_in_tree(tree):
            out.append((rel, lineno, name))
    return out


def reprompt_for_undefined_names(findings: list) -> str:
    """A targeted note listing EVERY undefined name at once (file:line) so ONE repair turn defines
    or imports them all — instead of the interpreter revealing one per run and the loop peeling a
    single layer per ~cycle (changing-failure-chain guidance, FF)."""
    items = "; ".join(f"`{name}` (used at {rel}:{lineno})" for rel, lineno, name in findings)
    n = len(findings)
    return (f"A static pass found {n} name{'s' if n != 1 else ''} used but defined NOWHERE and not "
            f"a builtin — each raises NameError when its line runs: {items}. Python reveals only the "
            "FIRST per run, so fix them ALL in this one pass (add the missing import, or define the "
            "function/variable) rather than one per attempt. Re-emit ALL files in full.")


def read_if_exists(project: Project, rel: str) -> str:
    p = project.code / rel
    return p.read_text() if p.is_file() else ""


def _signature(node) -> str:
    """One def/class signature line reconstructed from the AST, with NO body."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kw = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{kw}{node.name}({ast.unparse(node.args)}){ret}: ..."
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    return f"class {node.name}({bases}):" if bases else f"class {node.name}:"


def _doc1(node) -> str:
    """First non-blank line of a node's docstring as a trailing comment, or '' — just enough to
    convey intent without shipping the whole docstring."""
    doc = ast.get_docstring(node)
    first = next((l.strip() for l in (doc or "").splitlines() if l.strip()), "")
    return f"  # {first}" if first else ""


def api_digest(source: str) -> str:
    """An API SKELETON of one module: imports, top-level def/class signatures (+ a one-line
    docstring), class method signatures, and module-level constant NAMES — but NOT function bodies.

    The worker needs to CALL existing code correctly, not re-read its implementation; shipping whole
    source files is the dominant driver of context size — and thus KV-cache VRAM and prefill latency
    — on a memory-constrained local box (local-llm context-sizing guidance, MM). Reducing surrounding
    modules to signatures cut one real prompt from 41,858 -> 13,834 chars (-67%), which in turn
    halved the auto-sized num_ctx. Full bodies are shown ONLY for the file(s) the task is editing
    (see `package_api_digest`). Unparseable source falls back to its full text — never hide existing
    code from the model over a parse error."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.append(_signature(node) + _doc1(node))
            if isinstance(node, ast.ClassDef):
                lines += ["    " + _signature(s) for s in node.body
                          if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        elif isinstance(node, ast.Assign):
            lines += [f"{t.id} = ..." for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            lines.append(f"{node.target.id}: {ast.unparse(node.annotation)} = ...")
    return "\n".join(lines)


def package_api_digest(project: Project, full_bodies=()) -> str:
    """Existing package modules for the prompt: an `api_digest` (signatures only) for every file,
    EXCEPT those in `full_bodies` (the paths this task is editing), which are shown in FULL. The
    worker thus sees what it must call across the package, plus the complete body of only what it is
    changing — the structural prompt trim from the local-llm context-sizing guidance (MM). Replaces
    the old dump-every-source-file approach that drove prompts (and num_ctx, and VRAM) needlessly
    large."""
    pkg = project.code / project.package if project.package else project.code
    full = {str(p).lstrip("/") for p in full_bodies}
    chunks = []
    if pkg.is_dir():
        for f in sorted(pkg.rglob("*.py")):
            rel = f.relative_to(project.code).as_posix()
            src = f.read_text()
            chunks.append(f"=== EXISTING (full): {rel} ===\n{src}" if rel in full
                          else f"=== API: {rel} ===\n{api_digest(src)}")
    return "\n\n".join(chunks)


def write_files(project: Project, files: dict, allow_tests: bool,
                owners: dict = None, task_id: str = None) -> list:
    """Write the worker's emitted files into code/, enforcing two locks:
    - implementation tasks (allow_tests=False) may not write under tests/ at all;
    - a SHARED frozen file (conftest.py, the golden/constants module) has a single owning
      authoring task — every other task is refused, so a later P0.* run can't re-emit it in
      full and silently wipe an earlier run's fixtures (last-writer-wins)."""
    written = []
    code = project.code
    root_prefix = code.name + "/"     # e.g. "code/" — the build root's own dir name
    for rel, content in files.items():
        rel = rel.lstrip("/")
        # Defensive double-root strip: the worker is told paths are "relative to <root>/" and
        # some re-prefix that root, emitting code/pkg/x.py -> we'd then root it again under code/
        # and land at code/code/pkg/x.py. A leading `<root>/` is never legitimate (there is no
        # code/code/), so undo it; a no-op on correctly-rooted paths.
        while rel.startswith(root_prefix):
            rel = rel[len(root_prefix):]
            log(f"  stripped spurious root prefix from emitted path -> {rel}")
        if owners:
            o = owner_of(owners, rel)
            if o and o != task_id:
                log(f"  refusing to overwrite single-owner frozen file {rel} (owned by {o})")
                continue
        if not allow_tests and rel.startswith("tests/"):
            log(f"  refusing to write frozen test file: {rel}")
            continue
        dest = (code / rel).resolve()
        if code not in dest.parents and dest != code:
            log(f"  refusing path outside code/: {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = content if content.endswith("\n") else content + "\n"
        dest.write_text(text)
        written.append(rel)
        # Log the RESOLVED destination, not the worker's emitted string — a misrouted write
        # then shows in the log as where it actually landed, not where the model claimed.
        log(f"  wrote -> {dest}  ({len(text)} bytes, {text.count(chr(10))} lines)")
    return written


def summarize_pytest(output: str) -> str:
    lines = [l for l in output.splitlines() if l.strip()]
    summary = next((l for l in reversed(lines)
                    if re.search(r"\b\d+\s+(passed|failed|error|errors|skipped|"
                                 r"collected|deselected)\b|Interrupted|no tests ran", l)),
                   lines[-1] if lines else "")
    headers = [l.strip() for l in lines if l.startswith(("ERROR ", "FAILED ", "E   "))][:8]
    out = summary.strip("=! ").strip()
    if headers:
        out += "\n    " + "\n    ".join(headers)
    return out or "(no recognizable pytest summary)"


def skipped_count(output: str) -> int:
    """Number of skipped tests reported in a pytest run (0 if none). A green run with skips has
    demonstrated nothing about the skipped paths — and a skip-on-ImportError schism reports green
    while never running once, so a nonzero skip count on a PASS is a first-class signal, not noise."""
    m = re.search(r"\b(\d+)\s+skipped\b", output)
    return int(m.group(1)) if m else 0


def failure_signature(output: str) -> str:
    """The STABLE assertion/error lines of a failing pytest run — pytest's `E   ...` detail and
    `FAILED ...` short-summary lines, which carry the concrete value (`assert 0.0833 == 0.8`) but
    not volatile paths/timings. A signature that repeats UNCHANGED as the ladder climbs to a
    stronger model is the signature of a deterministic, correct computation against a WRONG expected
    value — an oracle bug in the frozen test, not a coding failure the worker can repair."""
    lines = [l.strip() for l in output.splitlines()
             if l.strip().startswith(("E   ", "FAILED ", "ERROR "))]
    return "\n".join(lines)


def normalize_pytest_cmd(cmd: str) -> str:
    """Run pytest through THIS interpreter so the check uses the exact Python/site-packages
    raster is running under, instead of a bare `pytest` that may not be on PATH."""
    return re.sub(r"(?<!-m )\bpytest\b", f"{sys.executable} -m pytest", cmd)


def _freeze_stub_env(stub_pkg: str) -> dict:
    """Env that loads the fallback-only product stub (raster/_freezestub.py) into a
    freeze-phase pytest collect, so tests importing the not-yet-built `stub_pkg` resolve
    instead of dying on ModuleNotFoundError. The stub is phase-scoped (set ONLY for these
    runs) and no-ops once the real package exists, so it can't mask a real broken import."""
    env = os.environ.copy()
    raster_root = str(Path(__file__).resolve().parent.parent)   # dir holding the `raster` pkg
    env["PYTHONPATH"] = os.pathsep.join(p for p in (raster_root, env.get("PYTHONPATH", "")) if p)
    env["PYTEST_ADDOPTS"] = (env.get("PYTEST_ADDOPTS", "") + " -p raster._freezestub").strip()
    env["RASTER_STUB_PACKAGE"] = stub_pkg
    return env


def run_test(project: Project, cmd: str, stub_pkg: str = None):
    """Run a pytest command in code/. `stub_pkg` (the product package) injects the
    freeze-phase absent-product stub — pass it ONLY for freeze collects (P0.* authoring
    and `--collect-only` freeze gates), never for an implementation gate."""
    env = _freeze_stub_env(stub_pkg) if stub_pkg else None
    try:
        proc = subprocess.run(cmd, shell=True, cwd=project.code, env=env,
                              capture_output=True, text=True, timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return False, out + f"\n[raster] test command timed out after {TEST_TIMEOUT}s"
    return proc.returncode == 0, (proc.stdout + proc.stderr)


def git_commit_push(project: Project, message: str) -> None:
    """Commit code/ + push to origin after a green task/gate. Best-effort: a git
    failure is logged but never changes the result. Non-PII identity, no co-author."""
    if not GIT_PUSH:
        return
    code = project.code
    if not (code / ".git").is_dir():
        log("  git: code/ is not a git repo — skipping")
        return
    cfg = project.cfg

    def g(*args):
        return subprocess.run(["git", "-C", str(code), *args], capture_output=True, text=True)
    try:
        g("add", "-A")
        if g("diff", "--cached", "--quiet").returncode == 0:
            log("  git: no changes to commit")
            return
        c = g("-c", f"user.name={cfg.git_author_name}",
              "-c", f"user.email={cfg.git_author_email}",
              "commit", "-m", message)        # no Co-Authored-By trailer
        if c.returncode != 0:
            log(f"  git: commit failed: {(c.stderr or c.stdout).strip()[:200]}")
            return
        log(f"  git: committed — {message.splitlines()[0]}")
        p = g("push", "origin", "HEAD")
        log("  git: pushed to origin" if p.returncode == 0
            else f"  git: push FAILED (committed locally): {(p.stderr or p.stdout).strip()[:200]}")
    except Exception as e:
        log(f"  git: error {e!r}")


# --------------------------------------------------------------------------- prompt
_AUTHOR_INSTRUCTIONS = (
    "You are AUTHORING FROZEN TESTS. Write thorough pytest tests with concrete, "
    "hand-computed golden values that assert real behavior (not stubs). The "
    "implementation does NOT exist yet — tests must currently FAIL, but must import "
    "and collect cleanly.\n"
    "CRITICAL — these golden values are FROZEN (implementation tasks cannot edit them), "
    "so they must be INTERNALLY CONSISTENT or they make a module unsatisfiable:\n"
    "- Any scalar count must equal the thing it counts (e.g. a declared total equals "
    "len() of the collection it describes).\n"
    "- Every derived value must be recomputed by hand from one stated formula and "
    "double-checked.\n"
    "- pytest.mark.parametrize: the number of names must equal the arity of each value row.\n"
    "Re-derive each number by hand and double-check it before emitting.\n"
    "LEAVE THE IMPLEMENTER NO LATITUDE — every gap you leave, a cheap local model fills "
    "from its training prior (and reasoning fills gaps more confidently, not more correctly). "
    "So pin, by asserting on them:\n"
    "- the data STRUCTURE the impl must produce — and, where a wrong shape is likely, assert "
    "against the anti-pattern too (e.g. a flat 1D sequence, NOT a 2D grid).\n"
    "- exact TYPES and FIELD NAMES (a dataclass with named fields, not 'a config dict').\n"
    "- formulas by NAME and DEFINITION with a worked example (e.g. Jaccard d=1-|a∩b|/|a∪b|, "
    "in [0,1], identical->0), so a bare 'distance' can't become a wrong metric.\n"
    "- the exact IDENTIFIER NAMES the downstream impl and any guard import. Expose golden "
    "values as importable MODULE-LEVEL CONSTANTS (e.g. GOLDEN_X = ...), NOT pytest fixtures — "
    "fixtures aren't importable, so a guard that introspects them finds nothing and passes "
    "VACUOUSLY.\n"
    "Where an EXTERNAL oracle exists for a frozen value (a conservation law, a type/dimension "
    "invariant, a domain rule the worker can get self-consistently wrong), also author a small "
    "IMPL-INDEPENDENT GUARD asserting that ground truth, under tests/golden/ (a path no task "
    "lists as a deliverable). It catches errors that are consistent in the tables yet wrong "
    "against the world.\n"
    "INVARIANT TESTS ARE NEGATIVE — PAIR EACH WITH A DIFFERENTIAL TEST. A conservation / "
    "determinism / 'mutating knob K doesn't break invariant I' test proves only that nothing got "
    "WORSE; it passes TRIVIALLY over a dead feature (absence is the most invariant-preserving state "
    "of all), so an invariant-only suite greens a no-op implementation. For EVERY parameter or "
    "input a module claims to consume, also author a DIFFERENTIAL test: two runs differing only in "
    "that knob's extremes must produce DIFFERENT observable output (e.g. history(tol=0) != "
    "history(tol=1)). Ship both — the invariant pins safety, the differential pins that the knob "
    "actually does something.\n"
    "CANONICAL NAMES, VOCABULARY, AND SIGNATURES — use ONE spelling per class/identifier across "
    "EVERY file (one name for the model class, one for the config class, etc.); ONE label/key "
    "VOCABULARY (don't key a golden dict by note-names in one file and index it by Roman numerals "
    "in another; don't mix 'major'/'maj' or differing status strings); and ONE call/constructor "
    "SIGNATURE for every shared object (build the model the same way everywhere — not "
    "Model(config) here and Model(n=…, b=…) there). The product is STUBBED during collection, so "
    "a stub resolves ANY name, key, or call shape — drift is invisible to the collect gate but "
    "makes the implementation unsatisfiable. Match DESIGN's canonical names/vocabulary/signatures "
    "exactly. Within a single file, never assert a thing two contradictory ways (e.g. chord names "
    "ARE note-names AND index them by Roman numeral) — that file is unsatisfiable on its own.\n"
    "SHARED FILES are SINGLE-OWNER — author conftest.py and any shared golden/constants module "
    "ONCE, in full, only in the task that owns them. Never re-emit a shared file from another "
    "module's authoring task: a later full re-emit silently wipes the fixtures an earlier run "
    "added (last-writer-wins). Emit only the files THIS task owns.\n"
    "NEVER SKIP ON ImportError — do NOT write `try: import <product>; except ImportError: "
    "pytest.skip(...)`. The product is STUBBED at collect time, so it always imports; that idiom "
    "only ever fires on a real NAME SCHISM (you imported a module name no task builds), and it "
    "turns that schism into a PERMANENT green that never runs — the worst false-green. Import "
    "product modules DIRECTLY, by their EXACT declared deliverable module name (if the deliverable "
    "is `pkg/chords.py`, import `pkg.chords` — never `pkg.chord`); a wrong name must fail loudly.\n"
    "DELEGATE, DON'T RE-STUB a pinned algorithm — when several tests need the same formula/metric, "
    "assert against the ONE canonical product symbol everywhere; never let a second call site get a "
    "fresh fake (a coarse tolerance lets a stub pass by luck and the real value is never checked).\n"
    "CONSTRUCT FRAMEWORK OBJECTS THE WAY THE FRAMEWORK ALLOWS — don't instantiate a framework "
    "subclass with a null/dummy collaborator its base class dereferences (e.g. a Mesa Agent with "
    "model=None, an ORM model with no session): that's an UNSATISFIABLE contract no correct impl "
    "can meet. Use a real or mock collaborator; if standalone pure-logic testing is the intent, the "
    "PRODUCT must expose an explicit null-collaborator path and the test must rely on that contract.\n"
    "ROUND-TRIP A FIXTURE THROUGH ITS OWN GOLDEN. When you hand-compute an expected value under "
    "ASSUMED PROPERTIES of a helper you also build from a golden table, the value and the helper can "
    "silently disagree — both look authoritative, yet no correct impl can satisfy the value. So add a "
    "3-line SELF-CHECK in the fixture that asserts the helper actually has the assumed properties "
    "BEFORE any value relies on them (a distance/metric adapter: assert m(x,x)==0 and m(a,b)==m(b,a)). "
    "It fails at authoring time instead of after the worker burns its budget on an unsatisfiable test.\n"
    "DON'T USE A HALF-MATRIX TABLE AS A METRIC. A symmetric relation stored de-duplicated (each "
    "unordered pair once, no diagonal) is a TABLE, not a FUNCTION: `GOLDEN.get((a,b), default)` keyed "
    "by free (a,b) misses the other order and the (a,a) diagonal, yielding an asymmetric, "
    "non-reflexive pseudo-metric. Either author the golden as a FULL symmetric table WITH diagonal "
    "(generate + consistency-check it once), or — where test independence isn't the point — call the "
    "ONE canonical product metric. A bespoke lambda over a half-matrix re-implements 'be a metric' "
    "and gets it wrong invisibly (the stubbed collect can't see it; impl can't satisfy it).\n"
    "DON'T ASSERT PER-STEP MONOTONICITY ON A STOCHASTIC PROCESS. Before writing a gate of the form "
    "'metric improves each step' — `np.all(np.diff(trend) >= 0)`, `np.sum(np.diff(trend) < -eps) <= "
    "N` — check the update rule you froze. If it's stochastic or non-greedy (random relocation, "
    "ε-exploration, simulated annealing), the observable only DRIFTS upward: it rises in expectation "
    "over a whole run, but individual step-to-step diffs go negative routinely, so per-step "
    "monotonicity tests for a BEST-IMPROVING dynamic you didn't build and a correct impl fails it. A "
    "small seed-average won't rescue it (a few seeds never smooth a random walk into monotonicity) "
    "and a hard dip-count cap just makes the verdict depend on which seeds you picked. Assert the "
    "property the process actually GUARANTEES: a robust NET rise over the run (trend[-1]-trend[0] >= "
    "a domain margin) PLUS up-steps OUTNUMBERING down-steps (np.sum(diff > eps) > np.sum(diff < "
    "-eps) — count-vs-count, never count-vs-constant), or a final-state-vs-random-baseline "
    "distribution test. Separate 'did the feature work' (the net/distributional claim — assert it) "
    "from 'is the trajectory shaped how I imagined' (the per-step picture in your head — don't). "
    "Then SEED-SMOKE-TEST it: re-run the gate with a disjoint seed list (seeds = [s + 100 for s in "
    "seeds]); if the verdict flips, the gate measures the seed list, not the model.\n"
    "DERIVE SHARED CONSTANTS, DON'T TRANSCRIBE THEM. A source-of-truth constant (a chord->pitch "
    "map, a rate table, an enum of statuses) must live in ONE canonical place; every consumer "
    "(renderer, exporter, UI, report) IMPORTS it — never carries a private copy 'based on' it. The "
    "moment two copies exist, a gate that asserts `consumer(x) == GOLDEN[x]` is comparing two "
    "transcriptions: it passes because they happen to agree, NOT because the consumer follows the "
    "source, and it stays green when the canonical value is later corrected (the edit it exists to "
    "catch). Keep exactly ONE independent hand-maintained oracle (the golden) to catch fat-finger "
    "errors in the source; wire the gate as (canonical)-vs-(golden), so the comparison is "
    "load-bearing. A `# based on <file>` comment over a literal is a duplication confession — import "
    "instead. PROVE the net bites: perturbing one entry of the canonical definition must turn the "
    "gate RED; if it stays green, the gate is wired to a duplicate. Likewise, never hardcode a magic "
    "number that equals a config field's default (bars*4 where 4 is cfg.beats_per_bar) — read the "
    "config, or the knob silently dies off-default.\n"
    "DON'T PIN A STRUCTURAL CONSTANT'S SIZE TO A RUN PARAMETER. A structural fact (how many chords "
    "the vocabulary DEFINES) and a run knob (how many a run SELECTS, n_chord_types) are DIFFERENT "
    "LAYERS; a single assertion speaks to only one. `len(DIATONIC_CHORDS) == 3`, where 3 is "
    "n_chord_types and the vocabulary has 7, is a category error, not a typo: no implementation can "
    "make a constant's length equal a per-run knob, so it's UNSATISFIABLE and survives escalation to "
    "a stronger model (the task is broken, not the worker). For every `len(CONSTANT) == <literal>`, "
    "ask which layer the number lives in; if that literal also appears as a config field / parameter "
    "value, you've probably conflated them. DERIVE the expected from the canonical source "
    "(`len(VOCAB) == len(diatonic_major())`), don't hand-type it: a typed expected is a transcription "
    "waiting to be wrong, and a derived one CANNOT be conflated with a knob. Reserve bare literals "
    "for genuinely axiomatic facts. And ISOLATE integrity/sanity guards from the real acceptance "
    "test: a bug in a decorative guard must not be able to fail a deliverable that actually passed, "
    "so give guards their own file/task or hold them to the same scrutiny as the contract beside them.\n"
    "EVERY DELIVERABLE MUST BE LOADED AND EXERCISED BY ITS OWN FROZEN TEST — THROUGH THE REAL "
    "LOADER, NOT RE-ENCODED INLINE — AND THE TEST MUST FAIL AT HEAD. For a data/config/asset "
    "deliverable (a configs/demo.yaml, a generated JSON/CSV), the test must locate it BY PATH "
    "(resolve the path FROM THE TEST FILE, not cwd, so it holds wherever pytest runs), load it via "
    "the production loader, and validate its contents. Building Config(**inline_dict) instead tests "
    "Config — which already exists and works — and says NOTHING about whether the artifact exists, "
    "loads, or is sane; an inline fixture is a SUBSTITUTE for the artifact, and a substitute is "
    "exactly what lets the artifact go missing. RED-BEFORE-GREEN: a deliverable task's frozen test "
    "must currently FAIL on the existing tree (artifact absent). A test already GREEN at HEAD is, by "
    "definition, not testing the deliverable — it is a false green, and (because the frozen test is "
    "the worker's only gradient toward the deliverable) a RUNAWAY-OUTPUT cause: with nothing pulling "
    "it toward the file, a weak model emits large, drifting, format-breaking text instead of the "
    "small artifact. Assert valid RANGES, not one pinned instance (the owner's domain intent stays at "
    "the human checkpoint). A non-`.py` deliverable whose filename never appears in its test is the "
    "tell — raster lint flags it.\n"
    "PIN THE OUTPUT SHAPE, NOT ONLY ITS VALUES. A COMPREHENSION miss — right numbers in the wrong "
    "CONTAINER — sails through value-only assertions and can hide behind any shallow error (a missing "
    "import, a typo) upstream of it, so the worker burns its whole budget peeling trivia and never "
    "reaches the structural defect. For any deliverable that PRODUCES ARTIFACTS (files, rows, records), "
    "assert the NUMBER and SHAPE explicitly, not just the contents: a parameter SWEEP emits exactly ONE "
    "summary artifact (assert it — `len(list(out.glob('*.csv'))) == 1`) with ONE ROW PER SWEPT VALUE "
    "(the swept value in a named column plus the aggregate observables), NOT one file per run dumping "
    "each run's raw per-step trajectory. State the file/row/column contract so a worker that built the "
    "wrong shape fails the frozen test instead of passing a naive eye."
)


def build_prompt(project: Project, module: dict, task: dict, authoring: bool) -> str:
    pkg = project.package
    deliverables = task.get("deliverables", [])
    # The files this task is editing get their FULL body in the prompt; every other package module
    # is reduced to an API digest (local-llm context-sizing guidance, MM) — the model must CALL them
    # correctly, not re-read their bodies, and the whole-package dump was the prompt's biggest term.
    editing = {str(d).lstrip("/") for d in deliverables}
    if not authoring:   # implementation tasks never write tests/ — don't even show them
        deliverables = [d for d in deliverables if not str(d).lstrip("/").startswith("tests/")]
    desc = f" ({project.description})" if project.description else ""
    parts = [
        f"You are an expert Python engineer building the {project.name} project{desc}.",
        output_contract(pkg, project.code.name),
        f"## Task {task['id']} — {task['title']}",
        f"Specification:\n{task['spec']}",
        f"Deliverables (paths relative to the {project.code.name}/ root, "
        f"NOT prefixed with {project.code.name}/): {deliverables}",
    ]

    if authoring:
        suffix = task["id"].split(".", 1)[1] if "." in task["id"] else ""
        target = module_by_id(project.spec, suffix) if suffix else None
        parts.append(_AUTHOR_INSTRUCTIONS)
        if target:
            behaviors = "\n".join(
                f"- {t['id']} {t['title']}: {t['spec']}"
                + (f"\n  test_notes: {t['test_notes']}" if t.get("test_notes") else "")
                for t in target.get("tasks", [])
            )
            parts.append(f"## Behaviors to cover (module {target['id']}):\n{behaviors}")
            gate = target.get("gate", {})
            if gate:
                parts.append(f"## Module gate to author ({gate.get('id')}): {gate.get('spec', '')}")
        existing = package_api_digest(project, editing)
        if existing:
            parts.append("## Existing package API (signatures for accurate imports/calls; full "
                         "body only for files you are editing):\n" + existing)
    else:
        test_file = task.get("unit_test", {}).get("file", "")
        frozen = read_if_exists(project, test_file)
        if frozen:
            parts.append(f"## FROZEN unit test you MUST satisfy ({test_file}) — do NOT modify it:\n{frozen}")
        existing = package_api_digest(project, editing)
        if existing:
            parts.append("## Existing package API (call these correctly; full body shown for the "
                         "file(s) you are editing):\n" + existing)
        parts.append("Write ONLY the implementation file(s) — never the test. Make the frozen test pass.")
    return "\n\n".join(parts)
