"""Unit tests for the generalized doer: prompt building, file parse/write guards,
pytest helpers, queue linearization, and build/test dry-runs — all offline (no
Ollama, no trundlr, no git)."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raster import execlib
from raster.config import Config
from raster.cli import main
from raster.queue import linearize
from raster.spec import Project, find_gate, find_task

SPEC = {
    "meta": {"package": "pkg", "project": "P",
             "workers": {"strong": "qwen", "worker": "llama"}, "think": False},
    "execution": {"resources": {"gpu": 2, "cpu": 3}, "ollama_host": "http://localhost:11434"},
    "modules": [
        {"id": "P0", "name": "test-infra", "tasks": [
            {"id": "P0.M0", "title": "Freeze M0 tests", "worker": "strong",
             "deliverables": ["tests/test_smoke.py"], "spec": "author frozen tests",
             "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest --collect-only -q tests/"}},
        ]},
        {"id": "M0", "name": "scaffold", "tasks": [
            {"id": "M0.T1", "title": "Package scaffold", "worker": "worker",
             "deliverables": ["pkg/__init__.py", "tests/test_smoke.py"], "spec": "make it import",
             "unit_test": {"file": "tests/test_smoke.py", "cmd": "pytest -q tests/test_smoke.py"}},
        ], "gate": {"id": "G0", "spec": "package imports",
                    "integration_test": {"file": "tests/gate.py", "cmd": "pytest -q tests/gate.py"}}},
    ],
}


def make_project(tmp_path) -> Project:
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    return Project(root=tmp_path, code=code, cfg=Config(),
                   ry={"project": "P", "package": "pkg"}, spec=SPEC)


# ------------------------------------------------------------------- execlib units
def test_parse_files():
    text = ("preamble\n=== FILE: pkg/a.py ===\nx = 1\n=== END FILE ===\n"
            "=== FILE: pkg/b.py ===\ny = 2\n=== END FILE ===\n")
    files = execlib.parse_files(text)
    assert files == {"pkg/a.py": "x = 1", "pkg/b.py": "y = 2"}


def test_write_files_guards(tmp_path):
    project = make_project(tmp_path)
    files = {"pkg/a.py": "x = 1", "tests/t.py": "frozen", "../escape.py": "nope"}
    written = execlib.write_files(project, files, allow_tests=False)
    assert written == ["pkg/a.py"]                      # tests/ refused, escape refused
    assert (project.code / "pkg" / "a.py").read_text() == "x = 1\n"
    assert not (project.code / "tests").exists()
    assert not (tmp_path / "escape.py").exists()
    # authoring may write tests/
    assert execlib.write_files(project, {"tests/t.py": "frozen"}, allow_tests=True) == ["tests/t.py"]


def test_normalize_pytest_cmd():
    out = execlib.normalize_pytest_cmd("pytest -q tests/")
    assert out == f"{sys.executable} -m pytest -q tests/"
    # idempotent: don't double-wrap
    assert execlib.normalize_pytest_cmd(out) == out


def test_summarize_pytest():
    s = execlib.summarize_pytest("...\n=== 2 failed, 45 passed in 1.2s ===\n")
    assert "2 failed, 45 passed" in s


def test_build_prompt_impl_filters_tests_and_names_package(tmp_path):
    project = make_project(tmp_path)
    (project.code / "tests").mkdir()
    (project.code / "tests" / "test_smoke.py").write_text("def test_x(): assert pkg")
    module, task = find_task(SPEC, "M0.T1")
    prompt = execlib.build_prompt(project, module, task, authoring=False)
    assert "building the P project" in prompt
    assert "pkg/example.py" in prompt                   # output contract uses the package
    assert "FROZEN unit test" in prompt and "def test_x" in prompt
    assert "tests/test_smoke.py" not in str(             # tests/ deliverable hidden from impl
        prompt.split("Deliverables")[1].splitlines()[0])


def test_build_prompt_authoring_lists_behaviors(tmp_path):
    project = make_project(tmp_path)
    module, task = find_task(SPEC, "P0.M0")
    prompt = execlib.build_prompt(project, module, task, authoring=True)
    assert "AUTHORING FROZEN TESTS" in prompt
    assert "Behaviors to cover (module M0)" in prompt and "M0.T1" in prompt
    assert "Module gate to author (G0)" in prompt


# ------------------------------------------------------------------- queue / find
def test_linearize_chain(tmp_path):
    project = make_project(tmp_path)
    chain = linearize(project, exec_cmd="raster")
    assert [c["id"] for c in chain] == ["P0.M0", "M0.T1", "G0"]
    assert chain[0]["command"] == "raster build P0.M0" and chain[0]["resource"] == 2
    assert chain[2]["command"] == "raster test G0" and chain[2]["resource"] == 3  # gate -> cpu


def test_find_gate_and_task():
    assert find_task(SPEC, "M0.T1")[1]["title"] == "Package scaffold"
    assert find_gate(SPEC, "G0")[1]["id"] == "G0"
    assert find_task(SPEC, "nope") == (None, None)


# ------------------------------------------------------------------- CLI dry-runs
def _on_disk_project(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    code = tmp_path / "code"
    (code / "designdocs").mkdir(parents=True)
    (code / "pkg").mkdir()
    (code / "designdocs" / "tasks.yaml").write_text(yaml.safe_dump(SPEC))
    (code / "raster.yaml").write_text(yaml.safe_dump({"project": "P", "package": "pkg"}))
    return tmp_path


def test_cli_build_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["build", "M0.T1", "--dir", str(root), "--dry-run"]) == 0
    assert "building the P project" in capsys.readouterr().out


def test_cli_test_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["test", "G0", "--dir", str(root), "--dry-run"]) == 0
    assert "gate G0" in capsys.readouterr().out


def test_cli_queue_dry_run(tmp_path, monkeypatch, capsys):
    root = _on_disk_project(tmp_path, monkeypatch)
    assert main(["queue", "--dir", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "raster build P0.M0" in out and "raster test G0" in out
