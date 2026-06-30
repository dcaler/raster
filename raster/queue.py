"""`raster queue` — linearize tasks.yaml into a single-parent trundlr task chain.

The two-level DAG (modules -> tasks + a gate) is flattened into one ordered chain:
per module, each task in order, then the module's gate; a failure auto-breaks
everything downstream (trundlr dependency_broken). Coding tasks run `raster build
<id>` on the GPU resource; gates run `raster test <id>` on the CPU resource.
"""

import os
import re

from raster import trundlr
from raster.build import resolve_rung, start_index
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


# Static run-time priors (hours) per (tier, think). One measured point —
# strong+think ≈ 1.75h (postIneq P0 authoring) — the rest are estimates, used only
# until `recent_run_hours` has real history of similar runs to replace them.
_RUN_HOURS = {
    ("worker", False): 0.25,
    ("worker", True):  0.5,
    ("strong", False): 0.5,
    ("strong", True):  1.75,
}


def recent_run_hours(worker: str, think: bool, history=None):
    """Median wall-clock of recent SIMILAR runs (same tier + think), or None when we
    have no history yet — the caller then falls back to the static prior. `history`
    is an iterable of {worker, think, hours} records (eventually harvested from
    trundlr's completed `raster:` tasks); None/empty means 'no data yet'."""
    samples = sorted(h["hours"] for h in (history or [])
                     if h.get("worker") == worker and bool(h.get("think")) == bool(think))
    return samples[len(samples) // 2] if samples else None


def estimate_hours(kind: str, worker: str, *, think: bool = False,
                   escalates: bool = False, history=None) -> float:
    if kind == "gate":
        return 0.1
    if kind == "checkpoint":
        return 1.0    # a manual human+claude review ~1h (and >0, which trundlr requires)
    # A strong-floored task starts think-OFF but is one climb from the think-on rung,
    # and the hard ones land there — so reserve it at the think-on cost up front.
    reasoning = bool(think or escalates)
    observed = recent_run_hours(worker, reasoning, history)
    if observed is not None:
        return observed
    return _RUN_HOURS.get((worker, reasoning), _RUN_HOURS[("strong", reasoning)])


def _think_budget(project, task: dict, authoring: bool):
    """(think_at_start, escalates_into_think) for a coding task. think_at_start: does
    the task's START rung reason (P0 authoring, a meta/per-task pin, or a think-on
    floor)? escalates_into_think: a think-OFF start whose very next rung flips think
    ON — one climb away (strong-floored impl), the case we budget at the think-on cost."""
    ladder = project.ladder()
    start = start_index(ladder, task.get("worker", "strong"))
    _, think_start = resolve_rung(ladder, start, task, authoring, project.meta)
    nxt = ladder[min(start + 1, len(ladder) - 1)]
    escalates = (not think_start) and bool(nxt["think"])
    return think_start, escalates


def _checkpoint_text(ck) -> str:
    """A checkpoint may be a bare prompt string or a {prompt/description, title} dict.
    The description IS the prompt — the literal instruction handed to the reviewer."""
    if isinstance(ck, dict):
        return (ck.get("prompt") or ck.get("description") or "").strip()
    return str(ck).strip()


def _checkpoint_item(cid: str, prompt: str, reviewers: list) -> dict:
    """A Layer-2 review checkpoint: a `command: null` task on the [human, claude]
    resources, sitting on a dependency edge. No automated runner claims those
    resources, so it stays `todo` and blocks everything downstream until a human
    marks it done — failing-closed with no special 'hold' primitive needed."""
    return {
        "id": cid,
        "title": f"raster: checkpoint {cid}",
        "description": prompt or f"Review checkpoint before {cid} — confirm nothing is haywire; end with go/no-go.",
        "command": None,                 # null command => no runner claims it => it blocks
        "resource": None,
        "resources": reviewers,
        "kind": "checkpoint",
        "duration": estimate_hours("checkpoint", ""),
    }


def _budgeted(duration: float, item: dict) -> float:
    """Reserve at least the task/gate's `budget:` (seconds) as the trundlr duration (hours),
    so a legitimately long gate (a GA/optimizer) gets a scheduling window matching its timeout."""
    b = item.get("budget")
    return max(duration, b / 3600.0) if b else duration


def _freeze_review_item(exec_cmd: str, cpu) -> dict:
    """The pre-queue gate node: runs once after all P0 authoring, before any implementation.
    `raster freeze-review` lints + EXECUTES red-before-green over the frozen suite and fails
    closed (non-zero => trundlr breaks the downstream chain), so the GPU build budget is spent
    only on a freeze that already cleared the cheap mechanical checks (freeze_review_gate UU/VV)."""
    return {
        "id": "freeze-review",
        "title": "raster: freeze-review",
        "description": "Pre-queue gate — lint + executed red-before-green over the frozen suite "
                       "(fails closed before any GPU build).",
        "command": f"{exec_cmd} freeze-review",
        "resource": cpu,
        "resources": [cpu],
        "kind": "gate",
        "duration": 0.25,
    }


def linearize(project, exec_cmd: str) -> list:
    res = project.execution.get("resources", {}) or {}
    gpu = res.get("gpu", project.cfg.gpu_resource)
    cpu = res.get("cpu", project.cfg.cpu_resource)
    reviewers = [r for r in (project.cfg.human_resource, project.cfg.claude_resource) if r]
    chain = []
    saw_authoring = False
    review_inserted = False
    for m in project.spec.get("modules", []) or []:
        # The freeze-review gate sits on the freeze->impl boundary: after every P0 authoring task,
        # before the first implementation module runs (and before that module's checkpoint).
        is_impl_module = not str(m.get("id", "")).startswith("P0")
        if saw_authoring and is_impl_module and not review_inserted:
            chain.append(_freeze_review_item(exec_cmd, cpu))
            review_inserted = True
        ck = m.get("checkpoint")
        if ck:                           # a review BEFORE this module's tasks run
            chain.append(_checkpoint_item(m["id"], _checkpoint_text(ck), reviewers))
        for t in m.get("tasks", []) or []:
            authoring = str(t.get("id", "")).startswith("P0")
            saw_authoring = saw_authoring or authoring
            think_start, escalates = _think_budget(project, t, authoring)
            chain.append({
                "id": t["id"],
                "title": f"raster: {t['id']}",
                "description": t.get("title", ""),
                "command": f"{exec_cmd} build {t['id']}",
                "resource": gpu,
                "resources": [gpu],
                "kind": "task",
                "duration": _budgeted(estimate_hours("task", t.get("worker", "strong"),
                                                     think=think_start, escalates=escalates), t),
            })
        g = m.get("gate")
        if g:
            chain.append({
                "id": g["id"],
                "title": f"raster: {g['id']}",
                "description": f"gate — {m.get('name', '')}",
                "command": f"{exec_cmd} test {g['id']}",
                "resource": cpu,
                "resources": [cpu],
                "kind": "gate",
                "duration": _budgeted(estimate_hours("gate", ""), g),
            })
    fck = project.spec.get("final_checkpoint")
    if fck:                              # a whole-system sign-off after the last module
        chain.append(_checkpoint_item("final", _checkpoint_text(fck), reviewers))
    return chain


def run_queue(args) -> int:
    project = load_project(args.dir)
    exec_cmd = args.exec_cmd or os.environ.get("RASTER_EXEC_CMD", "raster")
    chain = linearize(project, exec_cmd)
    if not chain:
        print("[raster queue] tasks.yaml has no modules yet — run `raster plan` first.")
        return 1

    checkpoints = [c for c in chain if c["kind"] == "checkpoint"]
    if checkpoints and not (project.cfg.human_resource and project.cfg.claude_resource):
        print("[raster queue] WARNING: this spec declares review checkpoints, but "
              "human_resource/claude_resource are unset in ~/.config/raster/config.toml.\n"
              "  Checkpoints will be queued with no reviewer resources — they still block "
              "downstream, but won't show up as anyone's task. Set both to fix.")

    if args.dry_run:
        total = sum(c["duration"] for c in chain)
        print(f"{len(chain)} tasks ({len(checkpoints)} checkpoint(s), ~{total:.2f}h), "
              f"exec_cmd={exec_cmd!r}:\n")
        for i, c in enumerate(chain):
            dep = chain[i - 1]["id"] if i else "—"
            resv = ",".join(map(str, c.get("resources") or [])) or "—"
            cmd = c["command"] or "MANUAL REVIEW (human+claude) — blocks until signed off"
            print(f"  {c['title']:26} res={resv:5}  {c['duration']:.2f}h  "
                  f"dep={dep:10}  [{cmd}]")
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
                "command": c["command"],          # null for checkpoints => no runner claims it
                "project_id": pid,
                "resource_ids": c.get("resources") or [c["resource"]],
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
