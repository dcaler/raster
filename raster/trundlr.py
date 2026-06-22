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


class TrundlrError(RuntimeError):
    """An API call failed; carries the server's response body (the 422 validation
    detail is far more useful than a bare 'Unprocessable Entity')."""


def _api(api_url: str, method: str, path: str, body=None, timeout: int = 30):
    url = f"{api_url.rstrip('/')}/api{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return None if resp.status == 204 else json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = (e.read() or b"").decode(errors="replace").strip()
        raise TrundlrError(f"{method} {url} -> HTTP {e.code} {e.reason}"
                           + (f"\n    {detail}" if detail else "")) from None


def set_project_directory(api_url: str, project_id: int, directory: str) -> None:
    """Point the trundlr project at the repo root (its `folder`) so queued commands run there."""
    _api(api_url, "PATCH", f"/projects/{project_id}", {"folder": directory})


def list_projects(api_url: str) -> list:
    return _api(api_url, "GET", "/projects/") or []


def create_project(api_url: str, name: str, folder: str = None, description: str = None) -> dict:
    body = {"name": name, "priority": 1}
    if folder:
        body["folder"] = folder
    if description:
        body["description"] = description
    return _api(api_url, "POST", "/projects/", body)


def resolve_project_id(api_url: str, name: str, folder: str = None,
                       description: str = None, create: bool = True):
    """Resolve a project NAME to trundlr's numeric id (trundlr keys projects by int id).
    Returns (id, created). Matches an existing project by exact name; creates one if absent
    and create=True, else returns (None, False)."""
    for p in list_projects(api_url):
        if p.get("name") == name:
            return int(p["id"]), False
    if not create:
        return None, False
    return int(create_project(api_url, name, folder, description)["id"]), True


def create_task(api_url: str, body: dict) -> dict:
    """Create one trundlr task (used by `raster queue` to chain the build)."""
    return _api(api_url, "POST", "/tasks/", body)


def update_task(api_url: str, task_id: int, body: dict) -> dict:
    """PATCH one existing trundlr task (e.g. re-budget its `duration`)."""
    return _api(api_url, "PATCH", f"/tasks/{task_id}", body)
