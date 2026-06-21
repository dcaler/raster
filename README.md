# raster

An **offline-first, test-driven code builder**. You write a design spec; raster
*rasterizes* it into a built, tested code repo — one deliverable per commit —
using local LLM workers via Ollama, orchestrated by trundlr.

> Part of the `ra*` family, alongside
> [rabbitHole](https://github.com/dcaler/rabbithole) (literature review) and
> [raconteur](https://github.com/dcaler/raconteur) (paper drafting). raster reads
> their output (a `litReview/` review, a `paper/` draft) when planning a build.

```
raster init           ▸ scaffold code/, its git repo, design-doc stubs; queue a plan task
raster plan           ▸ (interactive: you + Claude) author DESIGN.md + tasks.yaml   [next]
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

- scaffolds `code/` (raster works entirely inside it; the root is left alone),
- writes `code/raster.yaml` (build config, git-ignored) and `code/designdocs/`
  (`DESIGN.md`, `tasks.yaml`, `PROGRESS.md` + a `PLANNING.md` playbook),
- creates the project's own git repo named after it (`git init` + `gh repo create
  dcaler/<project>`, visibility asked per init) under the non-PII identity,
- queues an interactive **plan** task in trundlr assigned to **you + Claude**.

Then open a Claude session in the folder and run `raster plan` — it reads
`PLANNING.md`, absorbs the project's existing materials, and leads the design
interactively, producing `DESIGN.md` and `tasks.yaml`.

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
cost-of-error (see the planning playbook). `plan` is still an interactive stub.
