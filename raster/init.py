"""`raster init` — scaffold a project's code/ tree, its git repo, and design-doc
stubs, then optionally queue an interactive `plan` task in trundlr.

Run from the PROJECT ROOT (the shared working dir, alongside any litReview/ or
paper/). raster scaffolds and works entirely inside code/, never at the root.
The root is almost never empty and may lack code/ — both are fine.
"""

import re
import shutil
import subprocess
import sys
from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml

from raster.config import Config, load_config
from raster import trundlr


def log(msg: str) -> None:
    print(f"[raster init] {msg}", flush=True)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "package"


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
    else:
        log(f"gh: repo create failed ({(r.stderr or r.stdout).strip()[:160]}) — "
            f"repo exists locally; create the remote by hand")


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
    name = ask("Project name", default=prior.get("project") or root.name, preset=args.name)
    package = ask("Package / import name", default=prior.get("package") or slugify(name),
                  preset=args.package)
    description = ask("One-line description", default=prior.get("description", "").strip(),
                      preset=args.description)
    python_version = ask("Python version", default=prior.get("python", "3.11"),
                         preset=args.python)
    visibility = ask("Repo visibility (private/public)",
                     default=(prior.get("git", {}) or {}).get("visibility", "private"),
                     preset=args.visibility).lower()
    if visibility not in ("private", "public"):
        visibility = "private"
    tid_default = (prior.get("trundlr", {}) or {}).get("project_id", "")
    trundlr_id = ask("trundlr project id (blank to skip trundlr)",
                     default=tid_default, preset=args.trundlr_project_id)

    ctx = {
        "PROJECT": name,
        "PACKAGE": slugify(package),
        "DESCRIPTION": description or "(describe during raster plan)",
        "PYTHON_VERSION": python_version or "3.11",
        "TEST_RUNNER": "pytest",
        "GIT_HOST": cfg.git_host,
        "OWNER": cfg.git_owner,
        "VISIBILITY": visibility,
        "TRUNDLR_API": cfg.trundlr_api,
        "TRUNDLR_PROJECT_ID": trundlr_id or "null",
        "GPU": cfg.gpu_resource,
        "CPU": cfg.cpu_resource,
        "CALE": cfg.cale_resource,
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

    # ---- queue the interactive plan task (Cale + Claude) ----
    if not args.no_trundlr and trundlr_id:
        resources = [cfg.cale_resource, cfg.claude_resource]
        if not any(resources):
            log("trundlr: cale_resource/claude_resource are 0 in config — "
                "skipping plan task (set them to queue it)")
        else:
            try:
                t = trundlr.queue_plan_task(cfg.trundlr_api, int(trundlr_id),
                                            resources, name)
                log(f"trundlr: queued plan task #{t.get('id', '?')} "
                    f"(resources {[r for r in resources if r]})")
            except Exception as e:
                log(f"trundlr: could not queue plan task ({e!r}) — queue it by hand later")
    elif not trundlr_id:
        log("trundlr: no project id — skipped plan task")

    log("done.")
    print()
    print(f"  Scaffolded {name} in {code}")
    print( "  Next:  open a Claude session here and run `raster plan`")
    print( "         it will read code/designdocs/PLANNING.md and lead the design.")
    return 0
