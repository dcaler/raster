"""Minimal trundlr API client — enough for raster to queue a plan task.

trundlr is the orchestrator (the same one `raster build` will later submit the
doer task chain to). Here we only POST a single collaborative "plan" task.
"""

import json
import urllib.error
import urllib.request


def _api(api_url: str, method: str, path: str, body=None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api{path}",
        data=data, method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return None if resp.status == 204 else json.loads(resp.read())


def set_project_directory(api_url: str, project_id: int, directory: str) -> None:
    """Point the trundlr project at the repo root so queued commands run there."""
    _api(api_url, "PATCH", f"/projects/{project_id}", {"project_directory": directory})


def create_task(api_url: str, body: dict) -> dict:
    """Create one trundlr task (used by `raster queue` to chain the build)."""
    return _api(api_url, "POST", "/tasks/", body)


def queue_plan_task(api_url: str, project_id: int, resource_ids: list[int],
                    project_name: str) -> dict:
    """Create the interactive 'plan' task (Cale + Claude) for a project.

    The plan step is human+Claude collaborative design-doc authoring, so the task
    is assigned to both resources; its command points at `raster plan` and its
    description at the in-repo planning playbook.
    """
    body = {
        "title": f"raster: plan {project_name}",
        "description": ("Interactive design-doc authoring (Cale + Claude): complete "
                        "code/designdocs/DESIGN.md and tasks.yaml. Playbook: "
                        "code/designdocs/PLANNING.md."),
        "command": "raster plan",
        "project_id": int(project_id),
        "resource_ids": [r for r in resource_ids if r],
        "status": "todo",
    }
    return _api(api_url, "POST", "/tasks/", body)
