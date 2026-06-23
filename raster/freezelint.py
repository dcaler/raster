"""`raster lint` — a static cross-reference linter over the frozen test suite.

Promotes the freeze→impl defects a HUMAN caught (SchellingChords, 2026-06-23) that have a
mechanical oracle into a Layer-1 guard: each is checkable with ZERO implementation present
(pure AST over code/tests/, no imports, product stubbed at collect time anyway), and each
hard-blocks implementation while sailing through a green `--collect-only` gate. raster runs
this in the freeze gate so the scarce human checkpoint spends itself only on value- and
intent-correctness (which no linter can promote).

Checks (each maps to an observed defect class):
  * golden-key resolvability  — a literal subscript NAME["lit"] into a golden dict whose
    literal keys we can see must have "lit" among them (caught a note-name vs Roman schism).
  * fixture resolvability     — every fixture a test/fixture requests is defined somewhere
    (conftest or a test module) or is a pytest builtin (caught a fixture defined nowhere).
  * call-signature coherence  — a product symbol must not be called positionally in one file
    and by keyword in another (caught Model(config) vs Model(n_chord_types=…, …)).
"""

import ast

from raster.spec import load_project

# pytest's built-in fixtures — requested but never user-defined.
PYTEST_BUILTIN_FIXTURES = {
    "request", "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory", "monkeypatch",
    "capsys", "capsysbinary", "capfd", "capfdbinary", "caplog", "recwarn", "cache",
    "pytestconfig", "record_property", "record_testsuite_property", "doctest_namespace",
}


def _const_str(node):
    if isinstance(node, ast.Index):                 # py<3.9 slice wrapper (defensive)
        node = node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_fixture(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d   # @fixture(...) -> the fixture name
        name = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", None)
        if name == "fixture":
            return True
    return False


def _parametrized_names(fn: ast.FunctionDef) -> set:
    """Argument names introduced by @pytest.mark.parametrize — they look like fixtures in the
    signature but aren't, so the fixture check must exclude them."""
    names = set()
    for d in fn.decorator_list:
        if not isinstance(d, ast.Call):
            continue
        f = d.func
        if not (isinstance(f, ast.Attribute) and f.attr == "parametrize"):
            continue
        if d.args:
            spec = _const_str(d.args[0])
            if spec is not None:                        # "a,b" form
                names.update(n.strip() for n in spec.split(",") if n.strip())
            elif isinstance(d.args[0], (ast.List, ast.Tuple)):   # ["a","b"] form
                names.update(s for s in (_const_str(e) for e in d.args[0].elts) if s)
    return names


def _parse(files):
    trees = {}
    for f in files:
        try:
            trees[f] = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError as e:
            trees[f] = e
    return trees


def _golden_dicts(trees: dict) -> dict:
    """Module-level `NAME = {"k": ...}` dicts with all-string-literal keys -> {NAME: {keys}}."""
    out = {}
    for tree in trees.values():
        if isinstance(tree, SyntaxError):
            continue
        for node in tree.body:                          # module level only
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                keys = node.value.keys
                if keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                                for k in keys):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            out[t.id] = {k.value for k in keys}
    return out


def _product_symbols(trees: dict, package: str) -> set:
    syms = set()
    if not package:
        return syms
    for tree in trees.values():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and (node.module == package or node.module.startswith(package + "."))):
                for n in node.names:
                    syms.add(n.asname or n.name)
    return syms


def lint_frozen_tests(code, package: str) -> list:
    """Return a list of human-readable cross-reference violations (empty == clean)."""
    tests = code / "tests"
    files = sorted(tests.rglob("*.py")) if tests.is_dir() else []
    trees = _parse(files)
    violations = []

    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            violations.append(f"{f.name}:{tree.lineno}: SYNTAX ERROR — {tree.msg}")

    golden = _golden_dicts(trees)
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                name = node.value.id
                key = _const_str(node.slice)
                if name in golden and key is not None and key not in golden[name]:
                    violations.append(
                        f"{f.name}:{node.lineno}: {name}[{key!r}] — key absent from {name} "
                        f"(keys: {', '.join(sorted(golden[name])[:8])})")

    # fixtures: collect all definitions first, then check every request resolves
    defined = set(PYTEST_BUILTIN_FIXTURES)
    for tree in trees.values():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and _is_fixture(node):
                defined.add(node.name)
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not (node.name.startswith("test") or _is_fixture(node)):
                continue
            params = _parametrized_names(node)
            for a in node.args.args:
                if a.arg in ("self", "cls") or a.arg in params or a.arg in defined:
                    continue
                violations.append(f"{f.name}:{node.lineno}: {node.name}() requests fixture "
                                  f"{a.arg!r} defined nowhere")

    # call-signature coherence: a product symbol called positionally AND by keyword
    syms = _product_symbols(trees, package)
    shapes = {}                                         # sym -> {"pos":(f,line), "kw":(f,line,kwset)}
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in syms):
                npos = sum(not isinstance(a, ast.Starred) for a in node.args)
                kws = frozenset(k.arg for k in node.keywords if k.arg)
                s = shapes.setdefault(node.func.id, {})
                if npos >= 1 and not kws:
                    s.setdefault("pos", (f, node.lineno))
                elif npos == 0 and kws:
                    s.setdefault("kw", (f, node.lineno, kws))
    for sym, s in shapes.items():
        if "pos" in s and "kw" in s:
            pf, pl = s["pos"]
            kf, kl, kws = s["kw"]
            violations.append(f"{sym} called inconsistently: positional at {pf.name}:{pl} "
                              f"vs keyword({', '.join(sorted(kws))}) at {kf.name}:{kl}")
    return violations


def run_lint(args) -> int:
    project = load_project(args.dir)
    violations = lint_frozen_tests(project.code, project.package)
    if not violations:
        print("[raster lint] frozen-test cross-reference: clean")
        return 0
    print(f"[raster lint] {len(violations)} cross-reference violation(s) "
          f"(each hard-blocks implementation; the green collect can't see them):")
    for v in violations:
        print(f"  - {v}")
    return 1
