"""`raster plan` — interactive design-doc authoring (human + Claude).

NOT YET IMPLEMENTED as an automated command. Planning is an interactive Claude
session driven by the playbook `raster init` writes to code/designdocs/PLANNING.md.
For now this command points you there.
"""

from pathlib import Path


def run_plan(args) -> int:
    code = (Path(args.dir).resolve() if args.dir else Path.cwd()) / "code"
    playbook = code / "designdocs" / "PLANNING.md"
    print("raster plan is an interactive Claude+user step (not yet automated).")
    if playbook.is_file():
        print(f"\nOpen a Claude session in this folder and follow:\n  {playbook}")
        print("\nIt will read the project's materials, draft DESIGN.md and tasks.yaml,")
        print("and refine them with you interactively.")
    else:
        print("\nNo code/designdocs/PLANNING.md found — run `raster init` first.")
    return 0
