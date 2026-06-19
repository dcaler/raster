"""`raster build <task_id>` — run one CODING task.

The LLM implements (or, for P0.* tasks, authors frozen tests) against the task's
frozen unit test, in a bounded repair loop with worker->strong escalation; on a
green test the code is committed and pushed. This is the GPU/LLM half of the doer.
Invoked per-task by the trundlr runner (queued by `raster queue`), or by hand to
debug a single task.
"""

import os
import time

from raster import execlib, ollama
from raster.runlog import fmt_secs, log
from raster.spec import find_task, load_project

MAX_ATTEMPTS = int(os.environ.get("RASTER_MAX_ATTEMPTS", 4))
ESCALATE_AFTER = int(os.environ.get("RASTER_ESCALATE_AFTER", 2))   # worker -> strong
MAX_OUTPUT_CHARS = 6000                                            # of test output fed back


def run_build(args) -> int:
    project = load_project(args.dir)
    module, task = find_task(project.spec, args.task)
    if not task:
        log(f"build: task {args.task!r} not found in tasks.yaml")
        return 1

    authoring = args.task.startswith("P0")
    think = task.get("think", project.meta.get("think"))
    base_model = task.get("worker", "strong")
    model = project.model_for(base_model)
    unit_cmd = execlib.normalize_pytest_cmd(task["unit_test"]["cmd"])

    prompt = execlib.build_prompt(project, module, task, authoring)
    if args.dry_run:
        print(prompt)
        return 0

    max_attempts = args.max_attempts or MAX_ATTEMPTS
    host = project.ollama_host()
    log(f"START build={args.task} ({task['title']}) — "
        f"mode={'AUTHOR tests' if authoring else 'IMPLEMENT'}, model={model}, "
        f"max_attempts={max_attempts}")
    log(f"  ollama={host}  test_cmd={unit_cmd!r}  cwd={project.code}")
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, max_attempts + 1):
        active = model
        if base_model == "worker" and attempt > ESCALATE_AFTER:
            active = project.strong_model()
            log(f"attempt {attempt}/{max_attempts}: escalating worker -> {active}")
        log(f"=== attempt {attempt}/{max_attempts} (model={active}) ===")

        reply = ollama.chat(host, active, messages, label=f"{args.task} a{attempt}", think=think)
        files = execlib.parse_files(reply)
        if not files:
            log(f"attempt {attempt}: NO files parsed from {len(reply)}-char reply — re-prompting.")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             "No files detected. Re-emit using the exact "
                             "=== FILE: ... === / === END FILE === format."})
            continue

        log(f"attempt {attempt}: parsed {len(files)} file(s): {list(files)}")
        execlib.write_files(project, files, allow_tests=authoring)

        log(f"attempt {attempt}: running test: {unit_cmd}")
        t_test = time.monotonic()
        ok, output = execlib.run_test(project, unit_cmd)
        log(f"attempt {attempt}: test finished in {fmt_secs(time.monotonic() - t_test)} "
            f"-> {'PASS' if ok else 'FAIL'} | {execlib.summarize_pytest(output)}")
        if ok:
            kind = "author tests" if authoring else "implement"
            execlib.git_commit_push(project, f"{args.task} ({kind}): {task['title']}\n\n"
                                             f"Passed {unit_cmd!r} on attempt {attempt}.")
            log(f"DONE build={args.task}: PASS on attempt {attempt}")
            return 0

        if authoring:
            fix = ("These are pytest COLLECTION errors in the TEST files you wrote — the tests "
                   "must import and collect cleanly (they may still FAIL when run against the "
                   "not-yet-written implementation; that is expected and fine). Fix the TEST "
                   "files themselves and re-emit ALL files in full. Do NOT write implementation.")
        else:
            fix = "Fix the implementation and re-emit ALL files in full."
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
                         f"`{unit_cmd}` failed:\n\n{output[-MAX_OUTPUT_CHARS:]}\n\n" + fix})

    log(f"FAILED build={args.task} after {max_attempts} attempts. "
        f"Inspect {project.code} or re-run with more RASTER_MAX_ATTEMPTS.")
    return 1
