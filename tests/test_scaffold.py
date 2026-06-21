"""Scaffolding smoke test — `raster init` builds the expected tree, non-interactively,
with no git/remote/trundlr side effects."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raster.cli import build_parser
from raster.init import project_name_from_dir, run_init, slugify, render


def test_init_scaffolds_tree(tmp_path, monkeypatch):
    # isolate machine config so first-run defaults are used
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    args = build_parser().parse_args([
        "init", "--dir", str(tmp_path / "proj"),
        "--name", "WidgetForge", "--visibility", "private",
        "--no-git", "--no-remote", "--no-trundlr",
    ])
    assert run_init(args) == 0

    code = tmp_path / "proj" / "code"
    assert (code / "raster.yaml").is_file()
    assert (code / ".gitignore").is_file()
    assert (code / "widgetforge" / "__init__.py").is_file()   # import name = slug of project name
    for f in ("DESIGN.md", "tasks.yaml", "PROGRESS.md", "PLANNING.md"):
        assert (code / "designdocs" / f).is_file(), f

    # raster.yaml is valid YAML with the wizard values substituted
    cfg = yaml.safe_load((code / "raster.yaml").read_text())
    assert cfg["project"] == "WidgetForge"
    assert cfg["package"] == "widgetforge"   # always slugify(project name), never asked

    # tasks.yaml parses and carries the pre-filled meta block + empty modules
    spec = yaml.safe_load((code / "designdocs" / "tasks.yaml").read_text())
    assert spec["meta"]["project"] == "WidgetForge"
    assert spec["modules"] == []

    # no stray placeholder tokens left in any rendered file
    for f in code.rglob("*"):
        if f.is_file():
            assert "{{" not in f.read_text(), f


def test_slugify():
    assert slugify("Widget Forge!") == "widgetforge"
    assert slugify("123-ABC") == "123abc"


def test_init_stores_brief_and_feeds_plan(tmp_path, monkeypatch):
    # the long-form "what do you want to build" brief round-trips into raster.yaml (even
    # multi-line) and is surfaced in the planning playbook for the plan session.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    proj = tmp_path / "proj"
    brief = "Build a CLI that converts MIDI to abc notation.\nMust run fully offline."
    run_init(build_parser().parse_args(
        ["init", "--dir", str(proj), "--name", "midi2abc", "--brief", brief,
         "--no-git", "--no-remote", "--no-trundlr"]))
    code = proj / "code"
    ry = yaml.safe_load((code / "raster.yaml").read_text())
    assert ry["brief"] == brief                                   # multi-line preserved
    planning = (code / "designdocs" / "PLANNING.md").read_text()
    assert brief.splitlines()[0] in planning                      # fed into the plan playbook


def test_init_leaves_description_for_plan(tmp_path, monkeypatch):
    # init never asks for / sets a description; it leaves a placeholder for raster plan to fill.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    proj = tmp_path / "proj"
    run_init(build_parser().parse_args(
        ["init", "--dir", str(proj), "--name", "P", "--no-git", "--no-remote", "--no-trundlr"]))
    cfg = yaml.safe_load((proj / "code" / "raster.yaml").read_text())
    assert "generated during raster plan" in cfg["description"]


def test_project_name_from_dir_strips_datestamp():
    assert project_name_from_dir("260618_raster") == "raster"      # YYMMDD_
    assert project_name_from_dir("20260618_raster") == "raster"    # YYYYMMDD_
    assert project_name_from_dir("260618_my_project") == "my_project"  # only the prefix
    assert project_name_from_dir("raster") == "raster"             # no datestamp -> unchanged


def test_init_default_name_strips_datestamp(tmp_path, monkeypatch):
    # with no --name, the default project name is the dir name minus the datestamp prefix
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)   # non-interactive -> take defaults
    proj = tmp_path / "260618_widgets"
    run_init(build_parser().parse_args(
        ["init", "--dir", str(proj), "--no-git", "--no-remote", "--no-trundlr"]))
    cfg = yaml.safe_load((proj / "code" / "raster.yaml").read_text())
    assert cfg["project"] == "widgets"


def test_protect_keeps_authored_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    base = ["init", "--dir", str(tmp_path / "proj"), "--name", "P",
            "--no-git", "--no-remote", "--no-trundlr"]
    run_init(build_parser().parse_args(base))

    design = tmp_path / "proj" / "code" / "designdocs" / "DESIGN.md"
    design.write_text("# my authored design\n")
    run_init(build_parser().parse_args(base))          # re-init
    assert design.read_text() == "# my authored design\n"   # not clobbered
