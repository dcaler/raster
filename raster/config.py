"""Machine-level config for raster — ~/.config/raster/config.toml.

This is the PII boundary: personal/account details live here and never travel
into a project's committed files. Repos that raster builds are committed under a
deliberately non-PII identity (git.author_name/email) with no co-authorship.
Created with sensible defaults on first run.
"""

import os
from dataclasses import dataclass
from pathlib import Path

try:                                # stdlib on Python 3.11+
    import tomllib
except ModuleNotFoundError:         # 3.10 and older
    try:
        import tomli as tomllib    # type: ignore
    except ModuleNotFoundError:
        tomllib = None             # fall back to the tiny parser below


def _loads_toml(text: str) -> dict:
    """Parse raster's own simple config.toml when no TOML lib is available.
    Handles `[section]`, `key = "str" | int | true/false`, comments, and inline
    comments — sufficient for the flat schema raster writes (not general TOML)."""
    if tomllib is not None:
        return tomllib.loads(text)
    data: dict = {}
    section = data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = data.setdefault(line[1:-1].strip(), {})
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if val and val[0] in "\"'":          # quoted string (ignore trailing comment)
            q = val[0]
            val = val[1:val.index(q, 1)]
        else:
            val = val.split("#", 1)[0].strip()
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
        section[key.strip()] = val
    return data

DEFAULT_CONFIG_TOML = """\
# raster machine config. Personal details stay here — never committed into a project.

[ollama]
url    = "http://localhost:11434"
strong = "qwen3.6:27b-16k"   # conceptually tricky tasks + gate/test authoring
worker = "llama3.1:8b"       # scaffolding / boilerplate

[trundlr]
api_url      = "http://100.87.86.57:8251"
gpu_resource = 2             # runs LLM/doer tasks
cpu_resource = 3             # runs gate pytest
human_resource  = 0          # human reviewer's trundlr resource id — set this to queue plan tasks
claude_resource = 0          # Claude agent trundlr resource id — set this to queue plan tasks

[git]
host         = "github.com"
owner        = "dcaler"
author_name  = "raster"          # non-PII identity used for commits in BUILT repos
author_email = "raster@localhost"
co_authorship = false            # never emit Co-Authored-By trailers
"""


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "raster" / "config.toml"


@dataclass
class Config:
    ollama_url: str = "http://localhost:11434"
    strong_model: str = "qwen3.6:27b-16k"
    worker_model: str = "llama3.1:8b"
    trundlr_api: str = "http://100.87.86.57:8251"
    gpu_resource: int = 2
    cpu_resource: int = 3
    human_resource: int = 0
    claude_resource: int = 0
    git_host: str = "github.com"
    git_owner: str = "dcaler"
    git_author_name: str = "raster"
    git_author_email: str = "raster@localhost"
    co_authorship: bool = False


def load_config(create: bool = True) -> Config:
    """Load machine config, writing defaults on first run. Env vars override:
    OLLAMA_URL, RASTER_TRUNDLR_API."""
    p = config_path()
    if not p.exists():
        if create:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(DEFAULT_CONFIG_TOML)
        data = {}
    else:
        data = _loads_toml(p.read_text())

    o = data.get("ollama", {})
    t = data.get("trundlr", {})
    g = data.get("git", {})
    cfg = Config(
        ollama_url=o.get("url", Config.ollama_url),
        strong_model=o.get("strong", Config.strong_model),
        worker_model=o.get("worker", Config.worker_model),
        trundlr_api=t.get("api_url", Config.trundlr_api),
        gpu_resource=int(t.get("gpu_resource", Config.gpu_resource)),
        cpu_resource=int(t.get("cpu_resource", Config.cpu_resource)),
        human_resource=int(t.get("human_resource", Config.human_resource)),
        claude_resource=int(t.get("claude_resource", Config.claude_resource)),
        git_host=g.get("host", Config.git_host),
        git_owner=g.get("owner", Config.git_owner),
        git_author_name=g.get("author_name", Config.git_author_name),
        git_author_email=g.get("author_email", Config.git_author_email),
        co_authorship=bool(g.get("co_authorship", Config.co_authorship)),
    )
    # env overrides
    cfg.ollama_url = os.environ.get("OLLAMA_URL", cfg.ollama_url)
    cfg.trundlr_api = os.environ.get("RASTER_TRUNDLR_API", cfg.trundlr_api)
    return cfg
