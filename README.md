# raster

An **offline-first, test-driven code builder**. You write a design spec; raster
*rasterizes* it into a built, tested code repo — one deliverable per commit —
using local LLM workers via Ollama, orchestrated by trundlr.

> Part of the `ra*` family, alongside
> [rabbitHole](https://github.com/dcaler/rabbithole) (literature review) and
> [raconteur](https://github.com/dcaler/raconteur) (paper drafting). raster reads
> their output (a `litReview/` review, a `paper/` draft) when planning a build.

```
raster init           ▸ scaffold code/, its git repo, design-doc stubs (asks for a build brief)
raster plan           ▸ launch an interactive Claude session that authors DESIGN.md + tasks.yaml
raster queue          ▸ linearize tasks.yaml -> submit the trundlr build chain
raster build <id>     ▸ run one coding task (LLM implements/authors vs the frozen test)
raster test  <id>     ▸ run a gate or a unit-test assessment (no LLM)
```

raster lives at `github.com/dcaler/raster`. The repos it *builds* each push to
their own remote (`github.com/dcaler/<project>`).

---

## Install

```bash
git clone https://github.com/dcaler/raster.git
cd raster
pip install -e .
```

Requirements: Python ≥ 3.11, PyYAML. Optional: `gh` (to auto-create the project's
GitHub repo), a reachable trundlr API, and Ollama for the eventual build step.

## Machine setup

First run writes `~/.config/raster/config.toml` with defaults — Ollama models,
the trundlr API + resource ids (`human`, `claude`, gpu, cpu), and a **non-PII git
identity** used for commits in built repos (`co_authorship = false`). Personal
details live only here and never enter a project.

## Start a project

Run **from the project root** (the shared working dir — it may already hold a
`litReview/` or `paper/`, and need not have a `code/` yet):

```bash
cd ~/work/260618_myproject
raster init
```

`init`:

- asks for the project name and a long-form **build brief** ("what do you want to
  build today?"), stored in `raster.yaml` for the plan step,
- scaffolds `code/` (raster works entirely inside it; the root is left alone),
- writes `code/raster.yaml` (build config, git-ignored) and `code/designdocs/`
  (`DESIGN.md`, `tasks.yaml`, `PROGRESS.md` + a `PLANNING.md` playbook),
- creates the project's own git repo named after it (`git init` + `gh repo create
  dcaler/<project>`, visibility asked per init) under the non-PII identity.

Then run `raster plan` — it launches an interactive Claude session in the folder
that reads `PLANNING.md`, absorbs the brief and the project's existing materials,
and leads the design with you, producing `DESIGN.md` and `tasks.yaml`. (Use
`raster plan --no-launch` to just print the playbook path and drive it yourself.)

### What's committed vs. git-ignored

| Committed (shipped) | Git-ignored (build machinery) |
|---|---|
| `designdocs/DESIGN.md`, `PROGRESS.md`, `PLANNING.md` | `raster.yaml`, `designdocs/tasks.yaml` |
| the package under `code/<package>/` | local outputs, caches |

## Build a project

Once `tasks.yaml` has modules (from `raster plan`):

```bash
raster queue --dry-run          # preview the linearized chain
raster queue                    # submit it to trundlr (needs trundlr.project_id)

# or drive a single task/gate by hand (also what the trundlr runner invokes):
raster build M2.T2 --dry-run    # print the LLM prompt only
raster build M2.T2              # implement against the frozen test, commit on pass
raster test  G2                 # run a module gate
raster test  M2.T1              # re-run one task's unit test (assessment only)
```

`build`/`test` resolve paths, models, and the package name from `code/raster.yaml`
+ `tasks.yaml` (no hardcoding). Tuning knobs (env): `RASTER_MAX_ATTEMPTS`,
`RASTER_ESCALATE_AFTER`, `RASTER_TEST_TIMEOUT`, `RASTER_OLLAMA_TIMEOUT`,
`RASTER_GIT_PUSH=0` (skip commit/push), `RASTER_EXEC_CMD` (runner command for `queue`).

## Status

`init`, `queue`, `build`, and `test` are implemented (the generalized "doer":
frozen Phase-0 tests, bounded repair loop that climbs an **escalation ladder**
(`worker` → `strong` −think → `strong` +think — model *and* reasoning escalate
together, think-off-first / think-on-retry), commit-and-push on each green
task/gate). A task's `worker` is its starting tier *and* its floor — assign it by
cost-of-error (see the planning playbook). `plan` launches an interactive Claude
session seeded with the planning playbook (or `--no-launch` to drive it by hand).

---

## Guidance corpus (agent-facing)

Lessons distilled from real build failures — written for a future Claude working on
a raster/doer build, not for setup. **Match your symptom, then open the file.** Each
is self-contained and cross-linked; additions are lettered cumulatively (I…XX).

> **Start here:** [`freeze_review_gate_guidance.md`](./freeze_review_gate_guidance.md) — the meta file.
> raster concentrates correctness risk in one LLM-authored step (Phase-0 freeze); across a whole project
> the *costly* failures were frozen-layer defects, not implementation defects. This corpus is really a
> **freeze-review checklist** — run its four properties (red-before-green per deliverable, per-assertion
> satisfiability, producer-correspondence, sibling/gate assumption sweep) over the frozen tests *before*
> `queue`, and fall back to the symptom index below during a build.

**False greens — a test passes but proves nothing**
- [`false_green_guidance.md`](./false_green_guidance.md) — a green test you distrust: skip-on-`ImportError`, a re-stubbed algorithm passing by tolerance luck, an unsatisfiable framework built to never fail.
- [`dead_feature_false_green_guidance.md`](./dead_feature_false_green_guidance.md) — a delivered module/param is never imported; invariant-only "negative" tests that can't fail; a knob whose value never changes output.
- [`duplicated_constant_tautological_test_guidance.md`](./duplicated_constant_tautological_test_guidance.md) — a source-of-truth constant is copied into a consumer and the test compares copy-vs-copy; perturb the canonical source and the test stays green.
- [`deliverable_blind_test_guidance.md`](./deliverable_blind_test_guidance.md) — a build task's frozen test passes at HEAD before anything is written / never references its deliverable path (tested an inline dict, not the `configs/demo.yaml` it was supposed to produce); false green + runaway worker output. Enforce red-before-green.

**Frozen tests & oracle bugs — the test itself is wrong**
- [`frozen_test_infra_guidance.md`](./frozen_test_infra_guidance.md) — shared Phase-0 infra (conftest/golden) clobbered by a later task; names a later frozen test imports are missing.
- [`frozen_test_infra_guidance_addendum.md`](./frozen_test_infra_guidance_addendum.md) — *(addendum)* cross-reference lint; derive fixture constants from one source rather than duplicating.
- [`golden_oracle_consistency_guidance.md`](./golden_oracle_consistency_guidance.md) — a test hand-computes an expected value under assumed metric properties but builds the metric from a half-matrix golden; a worker plateaus on one exact wrong number.
- [`constant_vs_parameter_conflation_guidance.md`](./constant_vs_parameter_conflation_guidance.md) — `len(CONSTANT) == N` fails `assert 7 == 3` and N is also a config parameter; a buggy sanity-guard fails a deliverable that actually passed.
- [`oracle_bug_propagation_guidance.md`](./oracle_bug_propagation_guidance.md) — a frozen-test oracle bug reflects a wrong assumption the Phase-0 author applied across the *whole batch*, so fixing the one test that surfaced it leaves clones behind — most dangerously in the module **gate**, a separate file the per-task green never runs (M9.T1's `_step_count` spy recurred verbatim in G9's `gate_gui.py`). On every reconcile, grep the entire `tests/` tree (gates included) and fix all instances pre-emptively.

**Gate design — asserting the wrong property**
- [`stochastic_gate_sampling_guidance.md`](./stochastic_gate_sampling_guidance.md) — a gate asserts per-step monotonicity or a hard threshold on a noisy, seed-averaged metric; the verdict flips when you change the seed list.

**Doer / retry-loop mechanics**
- [`doer_write_path_robustness.md`](./doer_write_path_robustness.md) — the worker produced code but nothing reached disk; path/format mismatches; spurious `code/` prefix.
- [`doer_write_path_robustness_addendum.md`](./doer_write_path_robustness_addendum.md) — *(addendum)* "NO files parsed" with an opening `=== FILE:` but no terminator; what a format error costs; targeted re-prompt over spec-restate.
- [`retry_loop_context_economics_guidance.md`](./retry_loop_context_economics_guidance.md) — the prompt grows every retry; the loop escalates to a stronger model on an *identical* repeated failure; prefill/timeout blows up from bloated prompts.
- [`changing_failure_chain_guidance.md`](./changing_failure_chain_guidance.md) — the failure *changes* each attempt (`NameError: Path`→`datetime`→…); a sound test where the worker is progressing but runtime errors mask each other one-per-attempt, and a structural miss hides behind the chain. The mirror image of a plateau: escalate/give more turns, don't reconcile; add a static `pyflakes` pass.
- [`failure_chain_floor_guidance.md`](./failure_chain_floor_guidance.md) — the failure *count falls but levels off above zero* (19→18→17, then stuck): a **decaying plateau**. The worker fixed everything satisfiable and hit a floor of unsatisfiable oracle bugs. Read the asymptote, not the slope; test each residual for satisfiability and reconcile the unpassable. Three archetypes: fresh-fixture-per-case asserting accumulated state, a hand-authored golden no seed produces, a spy on a non-existent attribute. The needed correction to the changing-chain rule.
- [`local_llm_context_sizing_guidance.md`](./local_llm_context_sizing_guidance.md) — a "small" 8B model at 59 GB / 37% on CPU / ~0.6 tok/s: the KV cache is linear in the *context window* (32k default → ~32 GB), not the parameter count, so it overflows VRAM and spills to CPU. Pin `num_ctx` to the prompt; trim the prompt to API-signature digests (full body only for files being edited).

**Authoring & assignment**
- [`better_prompting_guidance.md`](./better_prompting_guidance.md) — writing worker task prompts and output contracts.
- [`task_Assignment_guidelines.md`](./task_Assignment_guidelines.md) — assigning tasks to model tiers by cost-of-error; what to freeze.

Recurring meta-lesson across the corpus: **read the trend of the error string, not just red/green.** A *same-failure plateau* (the worker returning the byte-identical wrong result across attempts, surviving model escalation) signals a broken task/oracle, not a weak model — stop and reconcile (a human "Cale+Claude" call), do **not** let auto-escalation spend the strong-model budget on it. A *changing failure chain* (a different error each attempt) is the mirror image: the test is sound and the worker is progressing, so escalate / give more turns — but watch for runtime errors masking each other and a structural miss hiding behind the chain (see `changing_failure_chain_guidance.md`). And a *decaying plateau* (the failure count falls but levels off above zero) is the trap between the two: early improvement looks like healthy progress, but the worker has fixed everything satisfiable and stalled on a floor of unsatisfiable oracle bugs (see `failure_chain_floor_guidance.md`). So the question is not just "does the error move?" but "what does it move *toward*?" — a chain heading to a nonzero asymptote is a plateau in disguise, and you test the residual for satisfiability rather than spending more strong-model budget on it.
