"""raster — offline-first, test-driven code builder in the ra* tool family.

raster rasterizes a design spec (designdocs/tasks.yaml) into a built, tested code
repo, task by task, using local LLM workers. Siblings: rabbitHole (literature
review) and raconteur (paper drafting).

Verbs:
  raster init    scaffold a project's code/ tree, git repo, and design-doc stubs
  raster plan    (interactive) author DESIGN.md + tasks.yaml  [not yet built]
  raster build   run each task/gate through the doer pipeline  [not yet built]
"""

__version__ = "0.1.0"
