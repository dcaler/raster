"""raster CLI — `raster <init|plan|build>`."""

import argparse

from raster import __version__


def _common(p):
    p.add_argument("--dir", help="project root (default: cwd)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="raster",
                                 description="Offline-first, test-driven code builder.")
    ap.add_argument("--version", action="version", version=f"raster {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="scaffold a project's code/ tree, repo, design-doc stubs")
    _common(init)
    init.add_argument("--name", help="project name (= repo name)")
    init.add_argument("--package", help="package / import name")
    init.add_argument("--description", help="one-line project description")
    init.add_argument("--python", help="target Python version (e.g. 3.11)")
    init.add_argument("--visibility", choices=["private", "public"],
                      help="GitHub repo visibility (asked if omitted)")
    init.add_argument("--trundlr-project-id", dest="trundlr_project_id",
                      help="existing trundlr project id (blank/omit to skip trundlr)")
    init.add_argument("--no-git", action="store_true", help="skip git init/commit")
    init.add_argument("--no-remote", action="store_true",
                      help="git init/commit locally but do NOT create/push the GitHub repo")
    init.add_argument("--no-trundlr", action="store_true", help="do not queue a plan task")

    plan = sub.add_parser("plan", help="(interactive) author DESIGN.md + tasks.yaml")
    _common(plan)

    queue = sub.add_parser("queue", help="linearize tasks.yaml -> submit the trundlr chain")
    _common(queue)
    queue.add_argument("--dry-run", action="store_true", help="print the chain, create nothing")
    queue.add_argument("--exec-cmd", help="command the runner invokes (default: raster)")

    build = sub.add_parser("build", help="run one coding task (LLM implements/authors)")
    _common(build)
    build.add_argument("task", help="task id, e.g. M2.T2 or P0.M2")
    build.add_argument("--dry-run", action="store_true", help="print the prompt and exit")
    build.add_argument("--max-attempts", type=int, default=0, help="override repair attempts")

    test = sub.add_parser("test", help="run a gate or unit-test assessment (no LLM)")
    _common(test)
    test.add_argument("id", help="gate id (e.g. G2) or task id (e.g. M2.T1)")
    test.add_argument("--dry-run", action="store_true", help="print the command and exit")

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "init":
        from raster.init import run_init
        return run_init(args)
    if args.cmd == "plan":
        from raster.plan import run_plan
        return run_plan(args)
    if args.cmd == "queue":
        from raster.queue import run_queue
        return run_queue(args)
    if args.cmd == "build":
        from raster.build import run_build
        return run_build(args)
    if args.cmd == "test":
        from raster.assess import run_assess
        return run_assess(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
