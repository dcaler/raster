"""`raster plan` — interactive design-doc authoring (human + Claude).

Launches an interactive Claude session seeded with a prompt that points it at the
in-repo planning playbook (`code/designdocs/PLANNING.md`). The session reads the
build brief + the project's materials, then drafts DESIGN.md and tasks.yaml with
you. `--no-launch` (or no `claude` on PATH) falls back to just printing the
playbook path so you can open a session by hand.
"""

import shutil
import subprocess
from pathlib import Path

PLAN_PROMPT = (
    "You are running the `raster plan` step. Read code/designdocs/PLANNING.md and follow it: "
    "absorb the build brief (code/raster.yaml `brief:`) and the project's existing materials "
    "(litReview/, paper/, and root files), then design code/designdocs/DESIGN.md and "
    "code/designdocs/tasks.yaml interactively with me, and write the generated one-line "
    "`description:` into code/raster.yaml. Start by reading PLANNING.md."
)


def _print_manual(playbook: Path, reason: str) -> None:
    print(reason)
    print(f"  {playbook}")
    print("It reads the brief + project materials, drafts DESIGN.md and tasks.yaml,")
    print("and refines them with you interactively.")


def run_plan(args) -> int:
    root = Path(args.dir).resolve() if args.dir else Path.cwd()
    playbook = root / "code" / "designdocs" / "PLANNING.md"
    if not playbook.is_file():
        print("No code/designdocs/PLANNING.md found — run `raster init` first.")
        return 1

    if getattr(args, "no_launch", False):
        _print_manual(playbook, "Open a Claude session in this folder and follow:")
        return 0
    if shutil.which("claude") is None:
        _print_manual(playbook, "`claude` is not on PATH — open a session yourself and follow:")
        return 0

    print(f"[raster plan] launching an interactive Claude session in {root} …")
    # Run from the project root so the session sees code/ AND the sibling litReview/ and
    # paper/ one level up. Inherits this terminal's stdio, so it's fully interactive;
    # returns control to the shell when you exit the session.
    return subprocess.run(["claude", PLAN_PROMPT], cwd=str(root)).returncode
