"""`raster queue` — linearize tasks.yaml into a single-parent trundlr task chain.

The two-level DAG (modules -> tasks + a gate) is flattened into one ordered chain:
per module, each task in order, then the module's gate; a failure auto-breaks
everything downstream (trundlr dependency_broken). Coding tasks run `raster build
<id>` on the GPU resource; gates run `raster test <id>` on the CPU resource.
"""

import os
import re

from raster import trundlr
from raster.spec import load_project


def _cache_project_id(project, pid: int) -> None:
    """Write the resolved numeric id back into code/raster.yaml (text-replace so comments
    survive), so the next `raster queue` is a direct submit with no name lookup."""
    ry = project.code / "raster.yaml"
    if not ry.is_file():
        return
    text = ry.read_text()
    # replace only the value, preserving indentation and any trailing comment
    new = re.sub(r"(?m)^(\s*project_id:\s*)[^#\n]*?(\s*(?:#.*)?)$", rf"\g<1>{pid}\g<2>", text, count=1)
    if new != text:
        ry.write_text(new)
        print(f"[raster queue] cached project id {pid} in code/raster.yaml")


def estimate_hours(kind: str, worker: str) -> float:
    if kind == "gate":
        return 0.1
    return 0.5 if worker == "strong" else 0.25


def linearize(project, exec_cmd: str) -> list:
    res = project.execution.get("resources", {}) or {}
    gpu = res.get("gpu", project.cfg.gpu_resource)
    cpu = res.get("cpu", project.cfg.cpu_resource)
    chain = []
    for m in project.spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            chain.append({
                "id": t["id"],
                "title": f"raster: {t['id']}",
                "description": t.get("title", ""),
                "command": f"{exec_cmd} build {t['id']}",
                "resource": gpu,
                "kind": "task",
                "duration": estimate_hours("task", t.get("worker", "strong")),
            })
        g = m.get("gate")
        if g:
            chain.append({
                "id": g["id"],
                "title": f"raster: {g['id']}",
                "description": f"gate — {m.get('name', '')}",
                "command": f"{exec_cmd} test {g['id']}",
                "resource": cpu,
                "kind": "gate",
                "duration": estimate_hours("gate", ""),
            })
    return chain


def run_queue(args) -> int:
    project = load_project(args.dir)
    exec_cmd = args.exec_cmd or os.environ.get("RASTER_EXEC_CMD", "raster")
    chain = linearize(project, exec_cmd)
    if not chain:
        print("[raster queue] tasks.yaml has no modules yet — run `raster plan` first.")
        return 1

    if args.dry_run:
        total = sum(c["duration"] for c in chain)
        print(f"{len(chain)} tasks (~{total:.2f}h), exec_cmd={exec_cmd!r}:\n")
        for i, c in enumerate(chain):
            dep = chain[i - 1]["id"] if i else "—"
            print(f"  {c['title']:24} res={c['resource']}  {c['duration']:.2f}h  "
                  f"dep={dep:10}  [{c['command']}]")
        return 0

    pid_raw = project.trundlr_project_id()
    if not pid_raw:
        print("[raster queue] no trundlr project id — set trundlr.project_id in code/raster.yaml.")
        return 1
    api = project.cfg.trundlr_api

    # trundlr keys projects by a NUMERIC id. The init default is the project NAME, so resolve
    # it to an id (creating the project if it doesn't exist yet) and cache the id for next time.
    if str(pid_raw).isdigit():
        pid = int(pid_raw)
    else:
        try:
            pid, created = trundlr.resolve_project_id(
                api, str(pid_raw), folder=str(project.root),
                description=project.description or None)
        except trundlr.TrundlrError as e:
            print(f"[raster queue] could not resolve trundlr project {pid_raw!r}: {e}")
            return 1
        print(f"[raster queue] {'created' if created else 'found'} trundlr project "
              f"{pid_raw!r} -> id {pid}")
        _cache_project_id(project, pid)

    try:
        trundlr.set_project_directory(api, pid, str(project.root))
        print(f"[raster queue] set project {pid} directory -> {project.root}")
    except Exception as e:
        print(f"[raster queue] warning: could not set project_directory: {e}")

    prev_id = None
    for c in chain:
        try:
            created = trundlr.create_task(api, {
                "title": c["title"],
                "description": c["description"],
                "command": c["command"],
                "project_id": pid,
                "resource_ids": [c["resource"]],
                "depends_on_id": prev_id,
                "duration": c["duration"],
                "status": "todo",
            })
        except trundlr.TrundlrError as e:
            where = f"after {prev_id}" if prev_id else "on the first task"
            print(f"[raster queue] FAILED creating {c['id']} ({where}): {e}")
            print(f"[raster queue] aborted — {'no' if not prev_id else 'a partial chain of'} "
                  f"tasks were created; fix the cause and re-run.")
            return 1
        prev_id = created["id"]
        print(f"[raster queue] created #{prev_id:<4} {c['title']}")

    print(f"[raster queue] done — {len(chain)} tasks chained under project {pid}")
    return 0
