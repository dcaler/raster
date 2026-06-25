"""Project loader + tasks.yaml helpers shared by queue / build / test.

A Project bundles the resolved paths, the machine config, raster.yaml, and the
build spec (designdocs/tasks.yaml) — generalizing the doer's hardcoded ROOT/CODE
and `schellingchords` package away into config the user controls.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from raster.config import Config, load_config


@dataclass
class Project:
    root: Path          # project root (the working dir; holds code/, maybe litReview/, paper/)
    code: Path          # root/code — raster works entirely in here
    cfg: Config         # machine config (git identity, ollama default, trundlr resources)
    ry: dict            # raster.yaml
    spec: dict          # designdocs/tasks.yaml

    @property
    def name(self) -> str:
        return self.ry.get("project") or self.meta.get("project") or self.root.name

    @property
    def package(self) -> str:
        return self.ry.get("package") or self.meta.get("package") or ""

    @property
    def description(self) -> str:
        return (self.ry.get("description") or "").strip()

    @property
    def meta(self) -> dict:
        return self.spec.get("meta", {}) or {}

    @property
    def execution(self) -> dict:
        return self.spec.get("execution", {}) or {}

    def trundlr_project_id(self):
        t = self.ry.get("trundlr", {}) or {}
        return t.get("project_id") or self.meta.get("trundlr_project_id")

    def ollama_host(self) -> str:
        # the runner sets OLLAMA_HOST to the bind address; prefer it, then spec, then config.
        return (os.environ.get("OLLAMA_HOST")
                or self.execution.get("ollama_host")
                or self.cfg.ollama_url)

    def model_for(self, worker_key: str) -> str:
        """Map a task's worker ('strong'/'worker') to a concrete model via meta.workers."""
        return (self.meta.get("workers", {}) or {}).get(worker_key, worker_key)

    def strong_model(self) -> str:
        return (self.meta.get("workers", {}) or {}).get("strong", self.cfg.strong_model)

    def ladder(self) -> list:
        """The ordered escalation ladder: a list of {worker, think} rungs that
        `raster build` climbs on repeated failure. A task starts on the rung
        matching its `worker` and climbs upward only — so a task's start tier is
        also its floor (it never escalates *down* to something cheaper). The
        default encodes worker→strong plus think-off-first / think-on-retry."""
        raw = self.meta.get("ladder")
        if not raw:
            return [dict(r) for r in DEFAULT_LADDER]
        return [{"worker": r.get("worker", "strong"), "think": bool(r.get("think", False))}
                for r in raw]


# Default escalation ladder (the doc's `[llama, qwen−think, qwen+think]`, generalized):
# cheapest worker first, then the strong model with reasoning off, then strong + reasoning
# on. Climbing flips BOTH the model and `think`, so reasoning is spent only where the cheap
# rungs have already failed against the frozen oracle.
DEFAULT_LADDER = [
    {"worker": "worker", "think": False},
    {"worker": "strong", "think": False},
    {"worker": "strong", "think": True},
]


def load_project(dir_arg=None) -> Project:
    root = Path(dir_arg).resolve() if dir_arg else Path.cwd()
    code = root / "code"
    spec_p = code / "designdocs" / "tasks.yaml"
    if not spec_p.is_file():
        raise SystemExit(f"[raster] no build spec at {spec_p} — run `raster init`/`raster plan` first")
    ry_p = code / "raster.yaml"
    ry = (yaml.safe_load(ry_p.read_text()) or {}) if ry_p.is_file() else {}
    spec = yaml.safe_load(spec_p.read_text()) or {}
    return Project(root=root, code=code, cfg=load_config(), ry=ry, spec=spec)


def find_task(spec: dict, task_id: str):
    for m in spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            if t.get("id") == task_id:
                return m, t
    return None, None


def find_gate(spec: dict, gate_id: str):
    for m in spec.get("modules", []) or []:
        g = m.get("gate")
        if g and g.get("id") == gate_id:
            return m, g
    return None, None


def module_by_id(spec: dict, mid: str):
    for m in spec.get("modules", []) or []:
        if m.get("id") == mid:
            return m
    return None


def authoring_owners(spec: dict) -> dict:
    """{frozen-test deliverable -> the single authoring (P0.*) task that owns it}.

    Every frozen file has ONE owning task = the first P0.* task in spec order that lists it
    as a deliverable. Shared infra (conftest.py, the golden/constants module) is otherwise
    re-emitted in full by every authoring run, so last-writer-wins silently clobbers earlier
    fixtures. `write_files` refuses any task that isn't a file's owner (see owner_of)."""
    owners = {}
    for m in spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            if not str(t.get("id", "")).startswith("P0"):
                continue
            for d in t.get("deliverables", []) or []:
                owners.setdefault(str(d).lstrip("/"), t["id"])
    return owners


def declared_modules(spec: dict, package: str) -> set:
    """The set of dotted PRODUCT module names any task declares as a deliverable, e.g.
    `pkg/chords.py` -> `pkg.chords`, `pkg/metrics/__init__.py` -> `pkg.metrics`, `pkg/__init__.py`
    -> `pkg`. Lets the linter prove a frozen test only imports modules some task actually builds —
    a test importing `pkg.chord` (singular) when the deliverable is `pkg.chords` (plural) is a
    name schism, not a pending feature, and must fail loudly rather than skip-on-ImportError."""
    mods = set()
    if not package:
        return mods
    for m in spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            for d in t.get("deliverables", []) or []:
                rel = str(d).lstrip("/")
                if not rel.endswith(".py"):
                    continue
                parts = rel[:-3].split("/")
                if parts and parts[-1] == "__init__":
                    parts = parts[:-1]
                if parts and parts[0] == package:
                    mods.add(".".join(parts))
    return mods


def lint_spec(spec: dict) -> list:
    """Static plan-validation (Layer-1, pre-run, zero implementation present): defects in
    tasks.yaml itself that statically guarantee a task can never satisfy its contract.

    * An IMPLEMENT (non-P0.*) task that lists a `tests/...` path as a DELIVERABLE: frozen
      test paths are an IMPLEMENT task's INPUT, never its output — `write_files` refuses every
      tests/ write from such a task, so the deliverable is unsatisfiable by construction. (A
      frozen test path belongs on the owning P0.* authoring task's deliverables, not here.)"""
    violations = []
    for m in spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            tid = str(t.get("id", ""))
            if tid.startswith("P0"):
                continue
            for d in t.get("deliverables", []) or []:
                if str(d).lstrip("/").startswith("tests/"):
                    violations.append(
                        f"task {tid}: deliverable {d!r} is under tests/ — an IMPLEMENT task "
                        f"cannot write frozen tests (they are its input). Move it to the owning "
                        f"P0.* authoring task's deliverables.")
    return violations


def owner_of(owners: dict, rel: str):
    """The owning task id for an emitted file `rel`, honoring directory deliverables (a
    declared path ending in '/' owns everything beneath it). None if unowned (free to write)."""
    rel = rel.lstrip("/")
    if rel in owners:
        return owners[rel]
    for d, owner in owners.items():
        if d.endswith("/") and rel.startswith(d):
            return owner
    return None
