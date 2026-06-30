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
from raster.spec import authoring_owners, find_task, load_project

MAX_ATTEMPTS = int(os.environ.get("RASTER_MAX_ATTEMPTS", 4))
ESCALATE_AFTER = int(os.environ.get("RASTER_ESCALATE_AFTER", 2))   # attempts per rung before climbing
MAX_OUTPUT_CHARS = 6000                                            # of test output fed back


def _missing_product_import(output: str, pkg: str) -> bool:
    """A freeze collect that still can't import the product package — the stub should have
    resolved it, so this is a configuration fault (stub not loaded), not a worker-repairable
    test bug. Looping a repair (least of all a thinking one) on it just burns the budget."""
    return bool(pkg) and (f"No module named '{pkg}'" in output
                          or f"No module named '{pkg}." in output)


def start_index(ladder: list, worker_key: str) -> int:
    """Where on the ladder a task begins = the first rung matching its `worker`.
    This is also its floor: escalation only climbs from here, never below."""
    for i, rung in enumerate(ladder):
        if rung["worker"] == worker_key:
            return i
    return 0


def rung_index(start: int, attempt: int, escalate_after: int, n: int) -> int:
    """Spend `escalate_after` attempts on the start rung, then climb one rung per
    further attempt (capped at the top). Turns the binary worker->strong jump into
    a graceful walk up an arbitrarily long ladder."""
    return min(start + max(0, attempt - escalate_after), n - 1)


def resolve_rung(ladder: list, idx: int, task: dict, authoring: bool, meta: dict):
    """Resolve a rung to (worker_key, think). `think` precedence: a per-task `think`
    pin wins; else P0 authoring is ALWAYS think-on (no oracle to repair against, so
    never gamble reasoning off there); else a global `meta.think` pin; else the rung."""
    worker_key = ladder[idx]["worker"]
    if "think" in task:
        think = bool(task["think"])
    elif authoring:
        think = True
    elif meta.get("think") is not None:
        think = bool(meta["think"])
    else:
        think = ladder[idx]["think"]
    return worker_key, think


def run_build(args) -> int:
    project = load_project(args.dir)
    module, task = find_task(project.spec, args.task)
    if not task:
        log(f"build: task {args.task!r} not found in tasks.yaml")
        return 1

    authoring = args.task.startswith("P0")
    ladder = project.ladder()
    start = start_index(ladder, task.get("worker", "strong"))
    unit_cmd = execlib.normalize_pytest_cmd(task["unit_test"]["cmd"])
    owners = authoring_owners(project.spec)   # single-owner write-protection on shared frozen infra
    stub_pkg = project.package if authoring else None   # freeze collects stub the absent product

    if args.dry_run:
        print(execlib.build_prompt(project, module, task, authoring))
        return 0

    max_attempts = args.max_attempts or MAX_ATTEMPTS
    host = project.ollama_host()
    start_key, _ = resolve_rung(ladder, start, task, authoring, project.meta)
    log(f"START build={args.task} ({task['title']}) — "
        f"mode={'AUTHOR tests' if authoring else 'IMPLEMENT'}, "
        f"start_rung={start + 1}/{len(ladder)} ({start_key}), max_attempts={max_attempts}")
    log(f"  ollama={host}  test_cmd={unit_cmd!r}  cwd={project.code}")

    prev_sig = None     # last attempt's failure signature, for the same-failure plateau abort
    chain_moved = False # did the failure signature CHANGE across attempts? (a moving chain, not a plateau)
    prev_count = None   # last attempt's FAILED count, to read the trajectory's slope (not just red/green)
    fail_counts = []    # the failed-count series across attempts — a decaying plateau levels off > 0
    feedback = None     # the SINGLE most-useful prior-failure summary — re-composed, never grown
    for attempt in range(1, max_attempts + 1):
        idx = rung_index(start, attempt, ESCALATE_AFTER, len(ladder))
        worker_key, think = resolve_rung(ladder, idx, task, authoring, project.meta)
        active = project.model_for(worker_key)
        rung = f"rung {idx + 1}/{len(ladder)} {worker_key} think={think}, model={active}"
        if idx > start:
            log(f"=== attempt {attempt}/{max_attempts}: escalated to {rung} ===")
        else:
            log(f"=== attempt {attempt}/{max_attempts} ({rung}) ===")

        # Re-compose a fixed, minimal prompt each attempt (the prompt is a resource you SPEND,
        # not a log you grow): task spec + frozen contract + current on-disk code (the latest
        # near-miss for the file(s) being edited, re-read fresh via package_api_digest; other modules
        # as signature-only API digests) + the SINGLE most-useful failure summary.
        # Escalation therefore inherits NO transcript — the slow/dear tier gets the smallest good
        # prompt, not the most polluted one (the two cost levers stop pulling against each other).
        prompt = execlib.build_prompt(project, module, task, authoring)
        messages = [{"role": "user", "content": prompt}]
        if feedback:
            messages.append({"role": "user", "content": feedback})

        reply = ollama.chat(host, active, messages, label=f"{args.task} a{attempt}", think=think)
        files = execlib.parse_files(reply)
        diag = execlib.parse_diagnostics(reply, files)
        if not files:
            # Self-diagnosing log (V) + targeted re-prompt (U): name the exact defect — an
            # opened-but-unterminated path or no marker at all — and the exact closer, so the
            # next ~16-min cycle is most likely to recover instead of re-drifting on a generic
            # 're-emit using the format' that re-sends the contract already in view.
            # A failed reply's body has NEGATIVE value here — re-feeding it anchors the model on
            # the same malformed block — so DROP it and carry only the short targeted note.
            log(f"attempt {attempt}: NO files parsed from {len(reply)}-char reply "
                f"({execlib.parse_failure_reason(diag)}) — re-prompting.")
            feedback = execlib.reprompt_for_parse_failure(diag, project.code.name)
            continue

        log(f"attempt {attempt}: parsed {len(files)} file(s): {list(files)}")
        if diag["unterminated"]:
            # Partial-parse silent drop (W): some blocks parsed, but an opening marker survived
            # with no `=== END FILE ===`, so that file is dropped from this write with no
            # re-prompt. The task would run a file short — surface it loudly (the failure that
            # emits SOME output is quieter and nastier than the one that emits none).
            log(f"  WARNING: {len(diag['unterminated'])} opened FILE block(s) had no "
                f"`=== END FILE ===` and were DROPPED from this write: {diag['unterminated']}. "
                f"If the test now fails on a missing file, that unterminated block is why.")
        execlib.write_files(project, files, allow_tests=authoring,
                            owners=owners, task_id=args.task)

        # Widen the feedback channel (FF): a runtime NameError chain (`Path`→`datetime`→a missing
        # helper) is revealed ONE defect per ~40-min test cycle, so four stacked undefined names
        # exhaust the budget before the real bug is reached. A sub-second static pass over the
        # just-written product code surfaces them ALL at once; folded into this attempt's failure
        # feedback, the next attempt fixes the whole chain in one turn. Conservative (flags only
        # names bound nowhere), so it only ever ADDS signal — it never red-lights a passing test.
        undef = [] if authoring else execlib.undefined_names(project)
        if undef:
            log(f"attempt {attempt}: static pass found {len(undef)} undefined name(s) "
                f"{sorted({u[2] for u in undef})} — each NameErrors when reached; surfacing the "
                f"FULL list so one repair fixes the chain, not one defect per ~cycle.")

        log(f"attempt {attempt}: running test: {unit_cmd}")
        t_test = time.monotonic()
        ok, output = execlib.run_test(project, unit_cmd, stub_pkg=stub_pkg)
        log(f"attempt {attempt}: test finished in {fmt_secs(time.monotonic() - t_test)} "
            f"-> {'PASS' if ok else 'FAIL'} | {execlib.summarize_pytest(output)}")
        if ok and not authoring and execlib.skipped_count(output):
            # A green IMPLEMENT test that skipped paths has demonstrated nothing about them —
            # a module-name schism skip-on-ImportError reports green while never running once.
            log(f"  WARNING: passed with {execlib.skipped_count(output)} SKIPPED test(s) — "
                f"those paths ran NOTHING; confirm it's not a name-schism false-green "
                f"(see `raster lint`).")
        if ok:
            kind = "author tests" if authoring else "implement"
            execlib.git_commit_push(project, f"{args.task} ({kind}): {task['title']}\n\n"
                                             f"Passed {unit_cmd!r} on attempt {attempt}.")
            log(f"DONE build={args.task}: PASS on attempt {attempt}")
            return 0

        if authoring and _missing_product_import(output, project.package):
            log(f"FAILED build={args.task}: a freeze-phase collect still cannot import the "
                f"product package {project.package!r} despite the fallback stub. That is a "
                f"configuration fault (the stub plugin did not load), not a worker-repairable "
                f"test bug — aborting rather than burning the repair budget on an unsatisfiable "
                f"loop. Check that `raster` is importable on PYTHONPATH for the test subprocess.")
            return 1

        if not authoring:
            # Same-failure plateau: two consecutive attempts produced the BYTE-IDENTICAL failing
            # value. A stronger model cannot satisfy an unsatisfiable task, so refuse to escalate
            # INTO it — this is the signature of a correct computation against a WRONG frozen
            # expected value (an oracle bug), not a coding failure the ladder can repair. Abort to a
            # HUMAN ORACLE CHECK now, before the expensive tier burns an attempt on a broken task.
            sig = execlib.failure_signature(output)
            failed = execlib.failed_count(output)
            fail_counts.append(failed)
            if sig and sig == prev_sig:
                span = "across an escalation" if idx > start else "on two consecutive attempts"
                log(f"FAILED build={args.task}: STABLE failing value {span} — the byte-identical "
                    f"failure repeated, so a stronger rung can't fix it. This is the signature of a "
                    f"correct computation against a WRONG expected value (an oracle bug in the frozen "
                    f"test), not a worker coding failure. Aborting to a HUMAN ORACLE CHECK rather than "
                    f"escalating into / burning the remaining attempts. "
                    f"Signature:\n    {sig.replace(chr(10), chr(10) + '    ')}")
                return 1
            if sig and prev_sig and sig != prev_sig:
                # The MIRROR image of the plateau (EE): the failure CHANGED, so the worker fixed the
                # last error and surfaced the next. But "did the error move?" is necessary, NOT
                # sufficient (failure-chain-floor guidance OO): read what it moves TOWARD. A changed
                # signature whose FAILED COUNT also dropped is genuine progress toward zero — escalate
                # / give more turns. A changed signature whose count did NOT drop (stuck or rising,
                # still > 0) is a DECAYING/oscillating PLATEAU: the worker is shuffling which residual
                # tests fail without converging, the fingerprint of an unsatisfiable oracle-bug floor.
                chain_moved = True
                if prev_count is not None and failed > 0 and failed >= prev_count:
                    log(f"attempt {attempt}: failure signature CHANGED but the FAILED COUNT did NOT "
                        f"drop ({prev_count} -> {failed}) — the chain is moving WITHOUT converging "
                        f"(a DECAYING PLATEAU). Read the asymptote, not the slope: this is likely a "
                        f"floor of UNSATISFIABLE residual tests (oracle bugs), not progress. If it "
                        f"holds, RECONCILE the residual — don't keep spending the strong tier on it.")
                else:
                    log(f"attempt {attempt}: failure signature CHANGED and the FAILED COUNT fell "
                        f"({prev_count} -> {failed}) — the worker is making real PROGRESS toward zero "
                        f"against a sound test. Escalating / giving more turns is correct here.")
            prev_sig = sig
            prev_count = failed

        # Logic failure: keep ONLY the latest failing output + a targeted fix as next attempt's
        # feedback (marginal value, latest only — the chain of prior near-misses is noise). The
        # near-miss CODE is carried on disk and re-read into the next prompt, not stuffed here.
        if authoring:
            fix = ("These are pytest COLLECTION errors in the TEST files you wrote — the tests "
                   "must import and collect cleanly (they may still FAIL when run against the "
                   "not-yet-written implementation; that is expected and fine). The product "
                   f"package ({project.package}) is STUBBED during this collect, so an import of "
                   "it is NOT the cause. Fix only SATISFIABLE test bugs: a pytest.mark.parametrize "
                   "name/value-arity mismatch, a name missing from the golden/constants module, a "
                   "bad import among the test files, or a syntax error. Re-emit ALL files in full. "
                   "Do NOT write implementation.")
        else:
            fix = "Fix the implementation and re-emit ALL files in full."
            if undef:
                # Lead with the WHOLE undefined-name list (FF) so the next attempt clears the entire
                # chain at once instead of the test revealing one NameError per ~cycle.
                fix = execlib.reprompt_for_undefined_names(undef) + "\n\n" + fix
        feedback = f"Your previous attempt failed `{unit_cmd}`:\n\n{output[-MAX_OUTPUT_CHARS:]}\n\n" + fix

    traj = "->".join(str(c) for c in fail_counts) if fail_counts else "n/a"
    leveled_off = len(fail_counts) >= 2 and fail_counts[-1] > 0 and fail_counts[-1] >= fail_counts[-2]
    if chain_moved and leveled_off:
        # DECAYING PLATEAU (failure-chain-floor guidance OO/PP): the chain MOVED but its failed count
        # leveled off ABOVE zero (failed[-1] >= failed[-2] > 0). The worker fixed everything that was
        # satisfiable and hit a floor of residual tests no implementation can pass — a subtler shape
        # of the same oracle-bug plateau the byte-identical check aborts on, masked early by healthy-
        # looking progress. More turns CANNOT cross it; the lever is reconcile, not budget.
        log(f"FAILED build={args.task} after {max_attempts} attempts — the failure count DECAYED then "
            f"LEVELED OFF above zero (failed count {traj}): a DECAYING PLATEAU. The worker fixed every "
            f"SATISFIABLE failure and hit a floor of residual tests no impl can pass — read the "
            f"ASYMPTOTE, not the slope. Do NOT re-queue with more turns (waste on a floor). RECONCILE: "
            f"test each residual failure for satisfiability (can ANY impl pass it given the rest of the "
            f"frozen suite?); a 'no' is an oracle bug — a HUMAN freeze call to fix the frozen test "
            f"(common floors: a fresh-fixture-per-case parametrize expecting accumulated state, a "
            f"hand-authored golden no seed produces, a spy on an attribute the product never defines).")
    elif chain_moved:
        # A MOVING chain still DESCENDING toward zero that exhausted the budget (GG): the worker WAS
        # progressing against a sound test and ran out of turns — often because runtime errors mask
        # each other (one revealed per ~cycle), so a flat MAX_ATTEMPTS under-budgets a deep error
        # stack. More attempts can be the right call. But FIRST read the asymptote (OO): a chain that
        # is heading toward a NONZERO floor (still ticking down by ones, 19->18->17) looks identical
        # to one heading to zero until it flattens — confirm the residual is genuinely satisfiable
        # before spending more of the strong tier on it. And rule out a STRUCTURAL miss (HH).
        log(f"FAILED build={args.task} after {max_attempts} attempts — the failure CHANGED and was "
            f"still DESCENDING (failed count {traj}): a MOVING chain, not a flat plateau. The worker "
            f"was progressing and ran out of turns. Before re-queuing, READ THE ASYMPTOTE: confirm it "
            f"is heading to ZERO, not converging on a nonzero floor — test each remaining failure for "
            f"satisfiability (a chain that ticks down by ones toward a residual of UNSATISFIABLE oracle "
            f"bugs looks like progress until it flattens). If satisfiable, RE-QUEUE with more "
            f"RASTER_MAX_ATTEMPTS or a higher floor tier; if not, RECONCILE the residual instead. "
            f"Either way hand-read the deliverable for a structural/comprehension miss (wrong output "
            f"SHAPE) that can hide behind a chain of shallow errors and never be reached in budget.")
    else:
        log(f"FAILED build={args.task} after {max_attempts} attempts (failed count {traj}). "
            f"Inspect {project.code} or re-run with more RASTER_MAX_ATTEMPTS.")
    return 1
