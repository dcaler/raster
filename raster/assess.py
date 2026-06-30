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
        budget = gate.get("budget")
    else:
        module, task = find_task(project.spec, tid)
        if not task:
            log(f"test: id {tid!r} not found as a gate or a task in tasks.yaml")
            return 1
        cmd = execlib.normalize_pytest_cmd(task["unit_test"]["cmd"])
        kind = "test"
        title = task.get("title", "")
        on_pass_msg = None        # a bare unit-test assessment doesn't advance the build
        budget = task.get("budget")

    if args.dry_run:
        print(f"[{kind} {tid}] cwd={project.code}\n{cmd}")
        return 0

    # A `--collect-only` gate is a freeze-phase structural check (it can't be an impl gate,
    # which must actually RUN), so it gets the absent-product stub; a real impl gate never does.
    stub_pkg = project.package if "--collect-only" in cmd else None

    log(f"START {kind}={tid} ({title!r:.80})")
    log(f"  cmd={cmd!r}  cwd={project.code}")
    t = time.monotonic()
    # a per-gate/task `budget:` (seconds) overrides the global timeout for a legitimately long
    # gate (e.g. a GA/optimizer over seeded simulation runs); see execlib.run_test.
    ok, output = execlib.run_test(project, cmd, stub_pkg=stub_pkg, timeout=budget)

    # A freeze gate (collect-only) also runs the Layer-1 cross-reference linter: it catches
    # mechanically-checkable freeze defects (unresolvable golden keys, undefined fixtures,
    # inconsistent call signatures) that a green collect can't see, before the human checkpoint.
    if stub_pkg and ok:
        from raster import freezelint
        lints = freezelint.lint_frozen_tests(project.code, project.package, project.spec)
        if lints:
            ok = False
            output += ("\n[raster] frozen-test cross-reference linter — "
                       f"{len(lints)} violation(s):\n" + "\n".join(f"  - {v}" for v in lints))

    log(f"{kind} {tid}: finished in {fmt_secs(time.monotonic() - t)} "
        f"-> {'PASS' if ok else 'FAIL'} | {execlib.summarize_pytest(output)}")
    skipped = execlib.skipped_count(output)
    if ok and not stub_pkg and skipped:
        # A green gate/test that SKIPPED tests has proven nothing about those paths — and a
        # skip-on-ImportError schism reports green while never running. Surface it loudly.
        log(f"  WARNING: {kind} {tid} passed with {skipped} SKIPPED test(s) — those paths ran "
            f"NOTHING. Confirm the skips are intentional, not a module-name schism masking a "
            f"false-green (see `raster lint`).")
    if not ok:
        # no repair loop here; surface the tail so the failure is self-contained.
        log(f"{kind} output tail:\n" + "\n".join(output.splitlines()[-25:]))
    elif on_pass_msg:
        execlib.git_commit_push(project, on_pass_msg)
    log(f"{'DONE' if ok else 'FAILED'} {kind}={tid}")
    return 0 if ok else 1
