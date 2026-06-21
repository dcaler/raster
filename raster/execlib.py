"""Shared executor guts for `raster build` (coding) and `raster test` (assessment):
prompt building, FILE-block parse/write, pytest running, and commit+push on pass.

Generalized from the SchellingChords doer — the package name, project name, and
project description are read from the Project (raster.yaml / tasks.yaml meta), not
hardcoded. Commits use the machine config's non-PII identity with NO co-authorship.
"""

import os
import re
import subprocess
import sys

from raster.runlog import log
from raster.spec import Project, module_by_id

FILE_RE = re.compile(r"=== FILE: (.+?) ===\n(.*?)\n=== END FILE ===", re.DOTALL)
TEST_TIMEOUT = int(os.environ.get("RASTER_TEST_TIMEOUT", 600))
GIT_PUSH = os.environ.get("RASTER_GIT_PUSH", "1") not in ("0", "false", "no", "")


def output_contract(pkg: str) -> str:
    example = f"{pkg}/example.py" if pkg else "example.py"
    return ("Output ONLY files, each wrapped EXACTLY like this (no prose, no markdown fences):\n\n"
            f"=== FILE: {example} ===\n"
            "<full file contents>\n"
            "=== END FILE ===\n\n"
            "Emit every file in full (not a diff). Paths are relative to the `code/` directory.")


def parse_files(text: str) -> dict:
    return {path.strip(): content for path, content in FILE_RE.findall(text)}


def read_if_exists(project: Project, rel: str) -> str:
    p = project.code / rel
    return p.read_text() if p.is_file() else ""


def package_sources(project: Project) -> str:
    """Current package sources — gives the model real signatures to call/extend."""
    pkg = project.code / project.package if project.package else project.code
    chunks = []
    if pkg.is_dir():
        for f in sorted(pkg.rglob("*.py")):
            rel = f.relative_to(project.code)
            chunks.append(f"=== EXISTING: {rel} ===\n{f.read_text()}")
    return "\n\n".join(chunks)


def write_files(project: Project, files: dict, allow_tests: bool) -> list:
    written = []
    code = project.code
    for rel, content in files.items():
        rel = rel.lstrip("/")
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
        log(f"  wrote {rel}  ({len(text)} bytes, {text.count(chr(10))} lines)")
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


def normalize_pytest_cmd(cmd: str) -> str:
    """Run pytest through THIS interpreter so the check uses the exact Python/site-packages
    raster is running under, instead of a bare `pytest` that may not be on PATH."""
    return re.sub(r"(?<!-m )\bpytest\b", f"{sys.executable} -m pytest", cmd)


def run_test(project: Project, cmd: str):
    try:
        proc = subprocess.run(cmd, shell=True, cwd=project.code,
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
    "against the world."
)


def build_prompt(project: Project, module: dict, task: dict, authoring: bool) -> str:
    pkg = project.package
    deliverables = task.get("deliverables", [])
    if not authoring:   # implementation tasks never write tests/ — don't even show them
        deliverables = [d for d in deliverables if not str(d).lstrip("/").startswith("tests/")]
    desc = f" ({project.description})" if project.description else ""
    parts = [
        f"You are an expert Python engineer building the {project.name} project{desc}.",
        output_contract(pkg),
        f"## Task {task['id']} — {task['title']}",
        f"Specification:\n{task['spec']}",
        f"Deliverables (paths under code/): {deliverables}",
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
        existing = package_sources(project)
        if existing:
            parts.append("## Current package sources (for accurate imports/signatures):\n" + existing)
    else:
        test_file = task.get("unit_test", {}).get("file", "")
        frozen = read_if_exists(project, test_file)
        if frozen:
            parts.append(f"## FROZEN unit test you MUST satisfy ({test_file}) — do NOT modify it:\n{frozen}")
        existing = package_sources(project)
        if existing:
            parts.append("## Current package sources (call these correctly):\n" + existing)
        parts.append("Write ONLY the implementation file(s) — never the test. Make the frozen test pass.")
    return "\n\n".join(parts)
