"""`raster test <id>` — run a GATE or a unit TEST assessment (no LLM).

`id` may be a module gate (e.g. G2 -> its integration_test) or a task (e.g. M2.T1
-> its unit_test). Same path resolution, timeout, and logging as the build path.
On a green gate the code is committed + pushed (a gate releases the next module);
plain test assessments don't commit. This is the CPU half of the doer.
"""

import time

from raster import execlib
from raster.runlog import fmt_secs, log
from raster.spec import find_gate, find_task, load_project


def run_assess(args) -> int:
    project = load_project(args.dir)
    tid = args.id

    module, gate = find_gate(project.spec, tid)
    if gate:
        cmd = execlib.normalize_pytest_cmd(gate["integration_test"]["cmd"])
        kind = "gate"
        title = gate.get("spec", module.get("name", ""))
        on_pass_msg = f"gate {tid} pass — {module.get('name', '')}".rstrip(" —")
    else:
        module, task = find_task(project.spec, tid)
        if not task:
            log(f"test: id {tid!r} not found as a gate or a task in tasks.yaml")
            return 1
        cmd = execlib.normalize_pytest_cmd(task["unit_test"]["cmd"])
        kind = "test"
        title = task.get("title", "")
        on_pass_msg = None        # a bare unit-test assessment doesn't advance the build

    if args.dry_run:
        print(f"[{kind} {tid}] cwd={project.code}\n{cmd}")
        return 0

    # A `--collect-only` gate is a freeze-phase structural check (it can't be an impl gate,
    # which must actually RUN), so it gets the absent-product stub; a real impl gate never does.
    stub_pkg = project.package if "--collect-only" in cmd else None

    log(f"START {kind}={tid} ({title!r:.80})")
    log(f"  cmd={cmd!r}  cwd={project.code}")
    t = time.monotonic()
    ok, output = execlib.run_test(project, cmd, stub_pkg=stub_pkg)
    log(f"{kind} {tid}: finished in {fmt_secs(time.monotonic() - t)} "
        f"-> {'PASS' if ok else 'FAIL'} | {execlib.summarize_pytest(output)}")
    if not ok:
        # no repair loop here; surface the tail so the failure is self-contained.
        log(f"{kind} output tail:\n" + "\n".join(output.splitlines()[-25:]))
    elif on_pass_msg:
        execlib.git_commit_push(project, on_pass_msg)
    log(f"{'DONE' if ok else 'FAILED'} {kind}={tid}")
    return 0 if ok else 1
