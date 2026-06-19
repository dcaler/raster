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
