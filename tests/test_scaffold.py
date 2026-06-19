"""Scaffolding smoke test — `raster init` builds the expected tree, non-interactively,
with no git/remote/trundlr side effects."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raster.cli import build_parser
from raster.init import run_init, slugify, render


def test_init_scaffolds_tree(tmp_path, monkeypatch):
    # isolate machine config so first-run defaults are used
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    args = build_parser().parse_args([
        "init", "--dir", str(tmp_path / "proj"),
        "--name", "WidgetForge", "--package", "widgetforge",
        "--description", "a thing that forges widgets",
        "--python", "3.11", "--visibility", "private",
        "--no-git", "--no-remote", "--no-trundlr",
    ])
    assert run_init(args) == 0

    code = tmp_path / "proj" / "code"
    assert (code / "raster.yaml").is_file()
    assert (code / ".gitignore").is_file()
    assert (code / "widgetforge" / "__init__.py").is_file()
    for f in ("DESIGN.md", "tasks.yaml", "PROGRESS.md", "PLANNING.md"):
        assert (code / "designdocs" / f).is_file(), f

    # raster.yaml is valid YAML with the wizard values substituted
    cfg = yaml.safe_load((code / "raster.yaml").read_text())
    assert cfg["project"] == "WidgetForge"
    assert cfg["package"] == "widgetforge"

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


def test_protect_keeps_authored_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    base = ["init", "--dir", str(tmp_path / "proj"), "--name", "P",
            "--description", "d", "--no-git", "--no-remote", "--no-trundlr"]
    run_init(build_parser().parse_args(base))

    design = tmp_path / "proj" / "code" / "designdocs" / "DESIGN.md"
    design.write_text("# my authored design\n")
    run_init(build_parser().parse_args(base))          # re-init
    assert design.read_text() == "# my authored design\n"   # not clobbered
