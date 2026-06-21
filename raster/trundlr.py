"""Minimal trundlr API client — enough for `raster queue` to submit the build chain.

trundlr is the orchestrator that runs the linearized `raster build`/`raster test`
tasks. This client sets a project's working directory and creates chained tasks.
"""

import json
import urllib.error
import urllib.request


def coerce_id(v):
    """A trundlr project id may be a numeric API id OR a project name (raster defaults
    it to the project name). Keep an all-digit id as an int; pass a name through as-is."""
    s = str(v)
    return int(s) if s.isdigit() else v


def _api(api_url: str, method: str, path: str, body=None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api{path}",
        data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return None if resp.status == 204 else json.loads(resp.read())


def set_project_directory(api_url: str, project_id, directory: str) -> None:
    """Point the trundlr project at the repo root so queued commands run there."""
    _api(api_url, "PATCH", f"/projects/{project_id}", {"project_directory": directory})


def create_task(api_url: str, body: dict) -> dict:
    """Create one trundlr task (used by `raster queue` to chain the build)."""
    return _api(api_url, "POST", "/tasks/", body)
