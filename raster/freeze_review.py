"""`raster freeze-review` — the pre-`queue` gate over the just-frozen test suite.

Across SchellingChords almost every COSTLY failure was a Phase-0 (frozen-layer) defect, not an
implementation defect — bad goldens and blind/unsatisfiable tests that the doer then spent hours
of GPU implementing against on the assumption the freeze was sound (freeze_review_gate guidance,
UU/VV/WW/XX). The cheapest place to catch them is the freeze; the most expensive is a multi-hour
build plateau. This command runs the four-property freeze checklist as a gate BEFORE any task is
queued — so the scarce human checkpoint and the GPU budget are spent only on what survives it.

Three of the four properties are mechanical and run here; the fourth is a reasoning pass the
command prints as a checklist for the human + Claude review (no linter can promote it):

  (1) Static cross-reference — the full `raster lint` suite (Layer-1 AST checks).
  (2) Red-before-green, per deliverable — EXECUTE each implement task's frozen unit test against
      the real tree at HEAD (product absent) and require it to FAIL. A test GREEN at HEAD is, by
      definition, not testing its deliverable — a false green and a runaway-output cause. (raster
      lint greps for the deliverable's path; this actually RUNS it — the zero-cost early exerciser.)
  (3) Gate red-before-green — the same, for every module GATE. Gates are the structural blind spot:
      authored with the unit tests but exercised only later, so they silently carry the same bugs
      (a green unit task says nothing about its gate). Exercise them at freeze.
  (4) Reasoning checklist — per-assertion satisfiability, producer-correspondence, and the
      sibling/gate assumption sweep; printed for the human+Claude pass, not auto-graded.
"""
from raster import execlib
from raster.freezelint import lint_violations
from raster.spec import load_project


def _iter_tasks(spec):
    for m in spec.get("modules", []) or []:
        for t in m.get("tasks", []) or []:
            yield m, t


def _redgreen(project, label, test_rel, cmd):
    """Run one frozen test against the REAL tree at HEAD (no product stub). Returns
    (status, detail): status in {'red','green','missing'}. 'green' is the defect — the test
    passes with the deliverable absent, so it can't be testing it."""
    if not test_rel or not (project.code / test_rel).is_file():
        return "missing", f"{label}: {test_rel or '(no test file)'} not authored yet"
    ok, output = execlib.run_test(project, execlib.normalize_pytest_cmd(cmd))   # stub_pkg=None -> real tree
    if ok:
        return "green", (f"{label}: {test_rel} is GREEN at HEAD with the deliverable absent — it is "
                         f"NOT testing its deliverable (false green; and, as the worker's only "
                         f"gradient, a runaway-output cause). Load the deliverable through the real "
                         f"loader and assert on it so it FAILS at HEAD. | {execlib.summarize_pytest(output)}")
    return "red", f"{label}: {test_rel} fails at HEAD (good — red-before-green holds)"


def run_freeze_review(args) -> int:
    project = load_project(args.dir)
    print(f"[raster freeze-review] {project.name} — Phase-0 freeze gate "
          f"(the defects that cost hours downstream when wrong)\n")

    blocking = []   # mechanical violations that should block `queue`

    # (1) static cross-reference
    static = lint_violations(project)
    print(f"(1) Static cross-reference (raster lint): "
          f"{'clean' if not static else str(len(static)) + ' violation(s)'}")
    for v in static:
        print(f"      - {v}")
    blocking += static

    # (2) red-before-green per implement task
    print("\n(2) Red-before-green — each deliverable's frozen test must FAIL at HEAD:")
    greens, missing, reds = [], 0, 0
    for _m, t in _iter_tasks(project.spec):
        tid = str(t.get("id", ""))
        if tid.startswith("P0"):                       # P0 authors tests; its own tests are red by design
            continue
        ut = t.get("unit_test") or {}
        status, detail = _redgreen(project, tid, ut.get("file", ""), ut.get("cmd", ""))
        if status == "green":
            greens.append(detail); print(f"      ✗ {detail}")
        elif status == "missing":
            missing += 1
        else:
            reds += 1
    if reds:
        print(f"      ✓ {reds} task test(s) correctly RED at HEAD")
    if missing:
        print(f"      … {missing} task test(s) not authored yet (Phase-0 freeze hasn't run)")
    blocking += greens

    # (3) gate red-before-green — the blind spot
    print("\n(3) Gate red-before-green — gates are exercised late, so they hide the same bugs:")
    g_greens, g_missing, g_reds = [], 0, 0
    for m in project.spec.get("modules", []) or []:
        g = m.get("gate")
        if not g:
            continue
        it = g.get("integration_test") or {}
        status, detail = _redgreen(project, str(g.get("id", "")), it.get("file", ""), it.get("cmd", ""))
        if status == "green":
            g_greens.append(detail); print(f"      ✗ {detail}")
        elif status == "missing":
            g_missing += 1
        else:
            g_reds += 1
    if g_reds:
        print(f"      ✓ {g_reds} gate(s) correctly RED at HEAD")
    if g_missing:
        print(f"      … {g_missing} gate(s) not authored yet")
    blocking += g_greens

    # (4) reasoning checklist — human + Claude, not mechanical
    print("\n(4) Reasoning pass (human + Claude — no linter promotes these). Before `queue`, confirm:")
    for line in (
        "satisfiability per assertion — can ANY faithful impl pass each one, given the rest of the suite?",
        "producer-correspondence — is each expected value DERIVED from the real producer, not hand-encoded in parallel?",
        "absolute thresholds are CALIBRATED, not guessed — a stochastic target pinned before measuring the frontier is the M9.T1 trap (prefer a differential/seed-averaged-margin claim, or measure a pilot and derive it).",
        "assumption sweep — grep the whole tests/ tree (gates included) for any bug you just reconciled; the same wrong belief recurs across siblings.",
    ):
        print(f"      [ ] {line}")

    print()
    if blocking:
        print(f"[raster freeze-review] {len(blocking)} BLOCKING violation(s) — fix before `raster queue` "
              f"(each is cheap now, a multi-hour build failure later).")
        return 1
    print("[raster freeze-review] mechanical checks clean. Complete the (4) reasoning pass, then `raster queue`.")
    return 0
