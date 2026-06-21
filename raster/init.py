"""`raster init` — scaffold a project's code/ tree, its git repo, and design-doc stubs.

Run from the PROJECT ROOT (the shared working dir, alongside any litReview/ or
paper/). raster scaffolds and works entirely inside code/, never at the root.
The root is almost never empty and may lack code/ — both are fine.
"""

import json
import re
import shutil
import subprocess
import sys
from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml

from raster.config import Config, load_config


def log(msg: str) -> None:
    print(f"[raster init] {msg}", flush=True)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "package"


def project_name_from_dir(dirname: str) -> str:
    """Guess a project name from the working-dir name the way the ra* family does:
    strip a leading {YYMMDD}_ (or {YYYYMMDD}_) datestamp prefix. e.g.
    '260618_raster' -> 'raster'. No datestamp -> the name unchanged."""
    return re.sub(r"^\d{6}(?:\d\d)?_", "", dirname) or dirname


def render(template_name: str, ctx: dict) -> str:
    text = (files("raster") / "templates" / template_name).read_text()
    for key, val in ctx.items():
        text = text.replace("{{" + key + "}}", str(val))
    return text


def ask(prompt: str, default=None, preset=None) -> str:
    """Prompt unless a preset (CLI arg) is given. Non-interactive -> default."""
    if preset is not None:
        return preset
    suffix = f" [{default}]" if default not in (None, "") else ""
    if not sys.stdin.isatty():
        return "" if default is None else str(default)
    try:
        resp = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        resp = ""
    return resp or ("" if default is None else str(default))


def ask_longform(prompt: str, preset=None) -> str:
    """Read a multi-line, free-form answer (the 'what do you want to build' brief).
    A preset (CLI arg) short-circuits; non-interactive -> empty. Interactively, the
    user types as much as they like and ends with Ctrl-D on a blank line."""
    if preset is not None:
        return preset
    if not sys.stdin.isatty():
        return ""
    print(f"  {prompt}")
    print("  (write as much as you like — the more the planner has to work with, the better;")
    print("   finish with Ctrl-D on a blank line)")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# ----------------------------------------------------------------- git / remote
def setup_git(code: Path, cfg: Config, ctx: dict, create_remote: bool) -> None:
    """git init code/ under a non-PII identity, scaffold commit (no co-authorship),
    and optionally create + push the project's own GitHub repo named after it."""
    if shutil.which("git") is None:
        log("git not found — skipping repo setup")
        return
    if not (code / ".git").is_dir():
        _run(["git", "init", "-b", "main"], cwd=code)
        log("git: initialized code/ (branch main)")
    _run(["git", "-C", str(code), "add", "-A"])
    if _run(["git", "-C", str(code), "diff", "--cached", "--quiet"]).returncode == 0:
        log("git: nothing to commit")
    else:
        # repo-local non-PII identity; NO Co-Authored-By trailer.
        c = _run(["git", "-C", str(code),
                  "-c", f"user.name={cfg.git_author_name}",
                  "-c", f"user.email={cfg.git_author_email}",
                  "commit", "-m", f"Scaffold {ctx['PROJECT']} (raster init)"], cwd=code)
        if c.returncode == 0:
            log("git: scaffold commit created")
        else:
            log(f"git: commit failed: {(c.stderr or c.stdout).strip()[:160]}")

    if not create_remote:
        return
    remotes = _run(["git", "-C", str(code), "remote"]).stdout.split()
    if "origin" in remotes:
        p = _run(["git", "-C", str(code), "push", "-u", "origin", "main"])
        log("git: pushed to existing origin" if p.returncode == 0
            else f"git: push failed: {(p.stderr or p.stdout).strip()[:160]}")
        return
    if shutil.which("gh") is None:
        log("gh not found — created repo locally only; add a remote manually")
        return
    vis = "--public" if ctx["VISIBILITY"] == "public" else "--private"
    slug = f"{cfg.git_owner}/{ctx['PROJECT']}"
    r = _run(["gh", "repo", "create", slug, vis,
              "--source", str(code), "--remote", "origin", "--push"])
    if r.returncode == 0:
        log(f"gh: created {vis.lstrip('-')} repo {slug} and pushed")
        return
    err = (r.stderr or r.stdout).strip()
    if "scope" in err.lower() or "createRepository" in err:
        hint = ("your gh token lacks the repo-creation scope — run "
                "`gh auth refresh -s repo` (or `-h github.com -s repo`), then re-run "
                "`raster init` to create + push the remote")
    elif "already exists" in err.lower() or "name already" in err.lower():
        hint = (f"{slug} already exists — add it as a remote: "
                f"`git -C {code} remote add origin git@{cfg.git_host}:{slug}.git && "
                f"git -C {code} push -u origin main`")
    else:
        hint = f"create {slug} on {cfg.git_host} and add it as origin by hand"
    log(f"gh: repo create failed — {hint}.")
    log(f"    (repo exists locally; gh said: {err[:140]})")


# ------------------------------------------------------------------- the command
def run_init(args) -> int:
    cfg = load_config()
    root = Path(args.dir).resolve() if args.dir else Path.cwd()
    code = root / "code"
    designdocs = code / "designdocs"
    existing = code / "raster.yaml"

    # reuse a prior trundlr id / name as defaults if re-initializing
    prior = {}
    if existing.is_file():
        try:
            prior = yaml.safe_load(existing.read_text()) or {}
        except Exception:
            prior = {}

    log(f"project root: {root}")
    name = ask("Project name", default=prior.get("project") or project_name_from_dir(root.name),
               preset=args.name)
    # long-form intent — the raw material `raster plan` designs from (stored in raster.yaml).
    brief = ask_longform("What do you want to build today?", preset=args.brief).strip()
    if not brief:
        brief = (prior.get("brief") or "").strip()   # keep a prior brief on re-init
    package = slugify(name)   # import name is always the slugified project name (not asked)
    # one-line description is NOT asked here — `raster plan` generates it from the detailed
    # design + planning session (it feeds the build LLM prompts), and writes it to raster.yaml.
    # A prior value (from a previous plan) is preserved across re-init.
    description = (prior.get("description") or "").strip()
    # python version is metadata only (tests run under raster's own interpreter), so it isn't
    # asked — default 3.11 (the shipped target); a prior value is preserved on re-init.
    python_version = prior.get("python") or "3.11"
    visibility = ask("Repo visibility (private/public)",
                     default=(prior.get("git", {}) or {}).get("visibility", "private"),
                     preset=args.visibility).lower()
    if visibility not in ("private", "public"):
        visibility = "private"
    # default the trundlr project id to the project name (same as project name itself
    # auto-fills from the dir); a prior id wins on re-init. Recorded in raster.yaml for
    # `raster queue` (the build chain) — init does not contact trundlr.
    tid_default = (prior.get("trundlr", {}) or {}).get("project_id") or name
    trundlr_id = ask("trundlr project id", default=tid_default, preset=args.trundlr_project_id)

    ctx = {
        "PROJECT": name,
        "PACKAGE": package,
        "BRIEF": brief or "(not provided at init — clarify with the user during planning)",
        "BRIEF_YAML": json.dumps(brief or "(not provided at init)"),
        "DESCRIPTION": description or "(to be generated during raster plan)",
        "PYTHON_VERSION": python_version or "3.11",
        "TEST_RUNNER": "pytest",
        "GIT_HOST": cfg.git_host,
        "OWNER": cfg.git_owner,
        "VISIBILITY": visibility,
        "TRUNDLR_API": cfg.trundlr_api,
        "TRUNDLR_PROJECT_ID": trundlr_id or "null",
        "GPU": cfg.gpu_resource,
        "CPU": cfg.cpu_resource,
        "HUMAN": cfg.human_resource,
        "CLAUDE": cfg.claude_resource,
        "STRONG_MODEL": cfg.strong_model,
        "WORKER_MODEL": cfg.worker_model,
        "OLLAMA_HOST": cfg.ollama_url,
        "DATE": date.today().isoformat(),
    }

    # ---- scaffold the tree (idempotent; never clobber authored design docs) ----
    designdocs.mkdir(parents=True, exist_ok=True)
    pkg_dir = code / ctx["PACKAGE"]
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_py = pkg_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text(f'"""{name} — see designdocs/DESIGN.md."""\n')

    def write(path: Path, template: str, protect: bool = False):
        if protect and path.exists() and path.read_text().strip():
            log(f"kept existing {path.relative_to(root)} (not overwritten)")
            return
        path.write_text(render(template, ctx))
        log(f"wrote {path.relative_to(root)}")

    write(code / "raster.yaml", "raster.yaml.tmpl")
    write(code / ".gitignore", "gitignore.tmpl")
    write(designdocs / "DESIGN.md", "DESIGN.md.tmpl", protect=True)
    write(designdocs / "tasks.yaml", "tasks.yaml.tmpl", protect=True)
    write(designdocs / "PROGRESS.md", "PROGRESS.md.tmpl", protect=True)
    write(designdocs / "PLANNING.md", "PLANNING.md.tmpl")

    # ---- git repo named after the project ----
    if not args.no_git:
        setup_git(code, cfg, ctx, create_remote=not args.no_remote)

    # The trundlr project id is recorded in raster.yaml for `raster queue` to submit the
    # BUILD chain later; init no longer queues an interactive plan task (raster plan launches
    # a Claude session directly, so there's nothing to queue).

    log("done.")
    print()
    print(f"  Scaffolded {name} in {code}")
    print( "  Next:  run `raster plan` — it launches an interactive Claude session that reads")
    print( "         code/designdocs/PLANNING.md and leads the design with you.")
    print( "         (use `raster plan --no-launch` to just print the playbook path instead.)")
    return 0
