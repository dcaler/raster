"""`raster lint` — a static cross-reference linter over the frozen test suite.

Promotes the freeze→impl defects a HUMAN caught (SchellingChords, 2026-06-23) that have a
mechanical oracle into a Layer-1 guard: each is checkable with ZERO implementation present
(pure AST over code/tests/, no imports, product stubbed at collect time anyway), and each
hard-blocks implementation while sailing through a green `--collect-only` gate. raster runs
this in the freeze gate so the scarce human checkpoint spends itself only on value- and
intent-correctness (which no linter can promote).

Checks (each maps to an observed defect class):
  * spec validity (lint_spec) — a tasks.yaml defect that's statically unsatisfiable, e.g. an
    IMPLEMENT task listing a frozen tests/ path as a deliverable (write_files refuses it).
  * skip-on-ImportError      — a frozen test that swallows an ImportError of a product module
    into pytest.skip turns a permanent NAME SCHISM into a permanent false-green (a whole module
    reports green while never running once). Flag the idiom; let the absent-product stub handle
    "not built yet" so a schism fails loudly instead.
  * module-import resolvability — every product module a frozen test imports must correspond to a
    declared deliverable module (caught `import chord` when the deliverable was `chords`).
  * dead-module reachability (lint_dead_modules) — a delivered product module that exists but is
    imported by no other product module is an island: dead code or a subsystem the consumer
    bypassed (reimplemented inline). A whole-system check; self-limits to modules already built.
  * copied source-of-truth constant (lint_copied_constants) — a consumer holding a PRIVATE copy of
    a canonical constant (a literal dict/list 'based on <module>', or the same constant name
    defined as a literal in 2+ product modules) makes its own contract gate tautological: it
    compares one copy to another and goes green even when the canonical source is edited. Derive,
    don't transcribe (caught sonify.py's private DIATONIC_CHORDS vs a golden copy). Whole-system.
  * golden-key resolvability  — a literal subscript NAME["lit"] into a golden dict whose
    literal keys we can see must have "lit" among them (caught a note-name vs Roman schism).
  * half-matrix lookup        — a golden pair-table stored de-duplicated (each unordered pair
    once, no diagonal) but used as a runtime lookup keyed by free (a,b) is a non-reflexive,
    asymmetric pseudo-metric: no correct impl can satisfy a value hand-computed under a REAL
    metric (caught segregation_index == 0.0833 vs an expected 0.8 over a half-matrix adapter).
  * stochastic per-step gate  — a per-step monotonicity / dip-count assertion on np.diff of a
    trend (`all(diff >= 0)`, `np.sum(diff < -eps) <= 1`) tests for a BEST-IMPROVING dynamic; on a
    stochastic/non-greedy update rule the observable only drifts upward, so the per-step claim is
    false and a small seed-average just makes the verdict seed-dependent (caught G5's `3 <= 1`). A
    smell the human adjudicates — flag it; the count-vs-count majority test is excluded.
  * fixture resolvability     — every fixture a test/fixture requests is defined somewhere
    (conftest or a test module) or is a pytest builtin (caught a fixture defined nowhere).
  * call-signature coherence  — a product symbol must not be called positionally in one file
    and by keyword in another (caught Model(config) vs Model(n_chord_types=…, …)).
"""

import ast
import re

from raster.spec import declared_modules, lint_spec, load_project

# comment phrases that confess a literal is a hand-synced COPY of a canonical constant elsewhere.
_COPY_MARKERS = ("based on", "copy of", "copied from", "keep in sync", "kept in sync",
                 "in sync with", "mirror of", "mirrors ", "duplicate of", "must match",
                 "matches ", "same as")

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


def _pair_tables(trees: dict) -> dict:
    """Module-level `NAME = {(x,y): ...}` dicts whose keys are ALL 2-tuples of string literals ->
    {NAME: {(x,y), ...}}. These are relation/distance tables — the half-matrix trap lives here."""
    out = {}
    for tree in trees.values():
        if isinstance(tree, SyntaxError):
            continue
        for node in tree.body:                          # module level only
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
                continue
            keys = node.value.keys
            pairs = set()
            ok = bool(keys)
            for k in keys:
                if (isinstance(k, ast.Tuple) and len(k.elts) == 2
                        and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                                for e in k.elts)):
                    pairs.add((k.elts[0].value, k.elts[1].value))
                else:
                    ok = False
                    break
            if ok:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = pairs
    return out


def _is_half_matrix(pairs: set) -> bool:
    """True if a symmetric relation is stored DE-DUPLICATED: some (x,y) lacks its reverse (y,x),
    or there's no diagonal (x,x) entry. Such a table is NOT a total metric — used as a runtime
    lookup keyed by arbitrary (a,b) it returns the default for the missing order and for (a,a)."""
    asymmetric = any((y, x) not in pairs for (x, y) in pairs if x != y)
    no_diagonal = not any(x == y for (x, y) in pairs)
    return asymmetric or no_diagonal


def _pair_lookup(node):
    """For `NAME[(a,b)]` and `NAME.get((a,b), …)` return (NAME, tuple_node); else (None, None)."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        sl = node.slice
        if isinstance(sl, ast.Index):                   # py<3.9 slice wrapper (defensive)
            sl = sl.value
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            return node.value.id, sl
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
          and node.func.attr == "get" and isinstance(node.func.value, ast.Name) and node.args
          and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2):
        return node.func.value.id, node.args[0]
    return None, None


# ordering / equality comparisons — the ops that make a per-step "sign of the difference" test.
_ORDER_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _is_diff_call(node) -> bool:
    """A discrete first-difference: `np.diff(x)` / `numpy.diff(x)` / `series.diff()` — anything
    `.diff(...)`. The per-step trend whose sign is the stochastic-monotonicity trap."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "diff")


def _diff_source_names(tree) -> set:
    """Names bound to a `np.diff(...)` result anywhere in the tree (`diffs = np.diff(avg_trend)`)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_diff_call(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _refs_diff(node, diff_names: set) -> bool:
    """`node` references a first-difference: a name bound to np.diff, or an inline np.diff(...)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in diff_names:
            return True
        if _is_diff_call(n):
            return True
    return False


def _reduction(node):
    """If `node` is a reduction CALL, return (name, [reduced exprs]); else (None, []). Handles both
    `np.all(EXPR)` / `sum(EXPR)` (function form -> args) and `(EXPR).all()` / `arr.sum()` (method
    form -> the receiver is the reduced expression)."""
    if not isinstance(node, ast.Call):
        return None, []
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr, [f.value, *node.args]        # receiver (EXPR in (EXPR).all()) + any args
    if isinstance(f, ast.Name):
        return f.id, list(node.args)
    return None, []


def _has_diff_sign_compare(exprs, diff_names: set) -> bool:
    """Any of `exprs` contains an ordering/equality Compare over a first-difference — a per-step
    SIGN test on the discrete differences (`diff > 0`, `diff < -eps`, `diff >= 0`)."""
    for e in exprs:
        for n in ast.walk(e):
            if (isinstance(n, ast.Compare) and any(isinstance(op, _ORDER_OPS) for op in n.ops)
                    and _refs_diff(n, diff_names)):
                return True
    return False


def _is_diff_sign_count(node, diff_names: set) -> bool:
    """A COUNT of per-step sign-violations: `np.sum(diff < -eps)` / `(diff < 0).sum()` /
    `np.count_nonzero(diff < 0)` — a sum/count reduction over a diff sign-comparison."""
    name, exprs = _reduction(node)
    return name in ("sum", "count_nonzero") and _has_diff_sign_compare(exprs, diff_names)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """An `except` clause that catches ImportError/ModuleNotFoundError (or bare except)."""
    t = handler.type
    if t is None:                                       # bare `except:`
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(getattr(n, "id", None) in ("ImportError", "ModuleNotFoundError") for n in names)


def _has_skip(nodes) -> bool:
    """A pytest.skip(...) / bare skip(...) / pytest.importorskip(...) call anywhere in `nodes`."""
    for body_node in nodes:
        for node in ast.walk(body_node):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in ("skip", "importorskip"):
                    return True
    return False


def _imported_package_modules(nodes, package: str) -> set:
    """Unambiguous product module imports under `package` among `nodes` (the try body):
    `import pkg.sub` and `from pkg.sub import ...` -> {"pkg.sub"}; also `from pkg import x`
    yields the candidate submodule "pkg.x" (ambiguous symbol-vs-module, reported in context)."""
    mods = set()
    for body_node in nodes:
        for node in ast.walk(body_node):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name == package or n.name.startswith(package + "."):
                        mods.add(n.name)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module == package:
                    for n in node.names:                # from pkg import sub  -> candidate pkg.sub
                        mods.add(f"{package}.{n.name}")
                elif node.module.startswith(package + "."):
                    mods.add(node.module)
    return mods


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


def _container_parts(rel_parts: tuple) -> tuple:
    """The dotted PACKAGE that contains a product file (for resolving relative imports):
    pkg/a.py -> ('pkg',), pkg/__init__.py -> ('pkg',), pkg/sub/b.py -> ('pkg','sub')."""
    parts = rel_parts[:-1] if rel_parts and rel_parts[-1] == "__init__" else rel_parts[:-1]
    return parts


def _product_imports(code, package: str) -> set:
    """Every product module referenced by an import anywhere in the package source (absolute or
    relative). Used to find ISLANDS: a delivered module imported by nothing is dead code or a
    subsystem the consumer silently bypassed. Over-marking reachable is safe (fewer false islands),
    so `from pkg import x` marks the candidate submodule pkg.x too."""
    referenced = set()
    pkg_root = code / package
    if not pkg_root.is_dir():
        return referenced
    for f in sorted(pkg_root.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        container = _container_parts(f.relative_to(code).with_suffix("").parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name == package or n.name.startswith(package + "."):
                        referenced.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    if node.module == package:
                        for n in node.names:
                            referenced.add(f"{package}.{n.name}")
                    elif node.module.startswith(package + "."):
                        referenced.add(node.module)
                        for n in node.names:
                            referenced.add(f"{node.module}.{n.name}")
                elif node.level > 0:                    # relative: resolve against the container
                    base = container[:len(container) - (node.level - 1)]
                    if not base or base[0] != package:
                        continue
                    if node.module:
                        referenced.add(".".join((*base, node.module)))
                        for n in node.names:
                            referenced.add(".".join((*base, node.module, n.name)))
                    else:
                        for n in node.names:            # from . import sibling
                            referenced.add(".".join((*base, n.name)))
    return referenced


def _module_file(code, dotted: str):
    rel = dotted.replace(".", "/")
    for cand in (code / (rel + ".py"), code / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _is_entrypoint(path) -> bool:
    """A module that's legitimately unimported by siblings: a __main__ module, or one with an
    `if __name__ == '__main__':` guard (a CLI / runnable entrypoint)."""
    if path.name == "__main__.py":
        return True
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"
                    and any(_const_str(c) == "__main__" for c in t.comparators)):
                return True
    return False


def lint_dead_modules(code, package: str, spec: dict) -> list:
    """Static reachability: a delivered PRODUCT module that exists on disk but is imported by no
    other product module (and isn't the package root or an entrypoint) is an ISLAND — dead code, or
    a subsystem the consumer reimplemented inline instead of calling (a dead-feature false-green:
    the island ships green over its own tests while wired to nothing). A whole-system / by-hand
    check — it self-limits to modules that already exist, so it's a no-op mid-freeze."""
    declared = declared_modules(spec, package) if spec else set()
    if not declared:
        return []
    referenced = _product_imports(code, package)
    violations = []
    for dotted in sorted(declared):
        if dotted == package:                           # the package root is legitimately unimported
            continue
        f = _module_file(code, dotted)
        if f is None or dotted in referenced or _is_entrypoint(f):
            continue
        violations.append(
            f"{dotted}: delivered product module is imported by NO other product module — an "
            f"island. Either dead code, or a subsystem the consumer bypassed (reimplemented inline "
            f"instead of importing it). Wire it into its consumer, or remove the deliverable.")
    return violations


def _assign_comment(lines: list, node: ast.Assign) -> str:
    """The comment text attached to an assignment: a trailing `# …` on its last line, plus a
    full-line `# …` directly above it (the two places a 'based on X' confession lands)."""
    parts = []
    end = getattr(node, "end_lineno", node.lineno)
    if 1 <= end <= len(lines) and "#" in lines[end - 1]:
        parts.append(lines[end - 1].split("#", 1)[1])
    i = node.lineno - 2                                  # 0-based line directly above the assign
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0 and lines[i].lstrip().startswith("#"):
        parts.append(lines[i].lstrip()[1:])
    return " ".join(parts)


def lint_copied_constants(code, package: str) -> list:
    """A source-of-truth constant copied into a CONSUMER is a tautology factory: a gate that
    asserts `render(x) == GOLDEN[x]` over a private product copy verifies that two transcriptions
    agree, not that the consumer follows the canonical source — it is structurally blind to an edit
    of that source (SchellingChords M6: sonify.py held a private DIATONIC_CHORDS 'based on chords.py'
    and its gate compared it to the golden; correcting a chord left both stale and the gate green).
    Two whole-system, product-source signals (no-op until the package is built):
      * a literal dict/list/set in a product module carrying a 'based on <other module>' style
        comment — the confession that a human promised to hand-sync two copies (DERIVE, don't copy);
      * the same collection-constant NAME defined as a literal in >=2 product modules — peer copies
        of one constant (make one canonical, import it in the others)."""
    pkg_root = code / package if package else code
    if not pkg_root.is_dir():
        return []
    stems = {f.stem for f in pkg_root.rglob("*.py")}
    by_name = {}                                         # const name -> {files defining it as literal}
    violations = []
    for f in sorted(pkg_root.rglob("*.py")):
        try:
            src = f.read_text()
            tree = ast.parse(src, filename=str(f))
        except (SyntaxError, OSError):
            continue
        lines = src.splitlines()
        for node in tree.body:                           # module level only
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.value, (ast.Dict, ast.List, ast.Set))):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)
                     and not (t.id.startswith("__") and t.id.endswith("__"))]
            if not names:
                continue
            for nm in names:
                by_name.setdefault(nm, set()).add(f.name)
            low = _assign_comment(lines, node).lower()
            marker = next((m for m in _COPY_MARKERS if m in low), None)
            if not marker:
                continue
            others = [s for s in stems if s != f.stem and re.search(rf"\b{re.escape(s)}\b", low)]
            if not (others or ".py" in low):             # must reference ANOTHER module's structure
                continue
            kind = type(node.value).__name__.lower()
            where = others[0] if others else "another module"
            violations.append(
                f"{f.name}:{node.lineno}: {'/'.join(names)} is a literal {kind} marked "
                f"'…{marker.strip()}…' referencing {where} — a hand-synced COPY of a canonical "
                f"constant. A gate comparing this copy to a golden tests TRANSCRIPTION AGREEMENT, "
                f"not truth, and stays green when the canonical source is edited. DERIVE it (import "
                f"the canonical definition); keep at most one independent test oracle.")
    for nm, files in sorted(by_name.items()):
        if len(files) >= 2:
            violations.append(
                f"{nm}: defined as a literal collection in {len(files)} product modules "
                f"({', '.join(sorted(files))}) — peer copies of one source-of-truth constant. "
                f"N mutually-agreeing copies enforce ZERO constraints; make one canonical and have "
                f"the other(s) import/derive from it (keep at most one independent test oracle).")
    return violations


def lint_frozen_tests(code, package: str, spec: dict = None) -> list:
    """Return a list of human-readable cross-reference violations (empty == clean). When `spec`
    is given, also checks that every product module a test imports is a declared deliverable."""
    tests = code / "tests"
    files = sorted(tests.rglob("*.py")) if tests.is_dir() else []
    trees = _parse(files)
    violations = []
    declared = declared_modules(spec, package) if spec else None

    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            violations.append(f"{f.name}:{tree.lineno}: SYNTAX ERROR — {tree.msg}")

    # skip-on-ImportError of a product module: turns a name schism into a permanent false-green.
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            mods = _imported_package_modules(node.body, package)
            if not mods:
                continue
            for h in node.handlers:
                if _catches_import_error(h) and _has_skip(h.body):
                    violations.append(
                        f"{f.name}:{h.lineno}: skip-on-ImportError of product import "
                        f"({', '.join(sorted(mods))}) — a name schism would skip (permanent "
                        f"false-green), not fail. Import directly; the freeze stub fabricates "
                        f"the not-yet-built module so a real schism fails loudly.")

    # module-import resolvability: a test importing a product module no task delivers.
    if declared is not None:
        for f, tree in trees.items():
            if isinstance(tree, SyntaxError):
                continue
            for node in ast.walk(tree):
                mods = set()
                if isinstance(node, ast.Import):
                    mods = {n.name for n in node.names
                            if n.name == package or n.name.startswith(package + ".")}
                elif (isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                      and node.module.startswith(package + ".")):
                    mods = {node.module}                # from pkg.sub import ... -> module is pkg.sub
                for mod in sorted(mods):
                    if mod != package and mod not in declared:
                        near = [d for d in declared if d.split(".")[-1].rstrip("s")
                                == mod.split(".")[-1].rstrip("s")]
                        hint = f" (did you mean {near[0]}?)" if near else ""
                        violations.append(
                            f"{f.name}:{node.lineno}: imports product module {mod!r} that no task "
                            f"declares as a deliverable — name schism, not a pending feature{hint}.")

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

    # half-matrix lookup: a de-duplicated symmetric relation (one order per pair, no diagonal)
    # used as a runtime lookup keyed by free (a, b) is a non-reflexive, asymmetric pseudo-metric.
    # A value hand-computed under a REAL metric can't be satisfied by it (SchellingChords M5.T1:
    # segregation_index returned 0.0833 under the half-matrix adapter, but expected 0.8).
    half = {n: p for n, p in _pair_tables(trees).items() if _is_half_matrix(p)}
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        for node in ast.walk(tree):
            name, keytup = _pair_lookup(node)
            if name in half and all(isinstance(e, ast.Name) for e in keytup.elts):
                violations.append(
                    f"{f.name}:{node.lineno}: {name}[(a,b)] looks up a HALF-MATRIX table by free "
                    f"variables — {name} stores each unordered pair once with no diagonal, so as a "
                    f"runtime (a,b) lookup it's asymmetric and non-reflexive (a pseudo-metric, not a "
                    f"metric). A value hand-computed under a real metric is then UNSATISFIABLE. "
                    f"Pre-expand {name} to a full symmetric table with diagonal, or wrap the lookup "
                    f"in a reflexive + two-order adapter.")

    # per-step monotonicity / dip-count threshold on np.diff of a (likely stochastic) trend.
    # Asserting the metric improves at (almost) EVERY step is a BEST-IMPROVING signature; on a
    # stochastic / non-greedy update rule (faithful random Schelling relocation) the observable only
    # DRIFTS upward and per-step dips are expected, so the assertion is false against a correct impl
    # (SchellingChords G5: `np.sum(diff < -1e-6) <= 1` failed `3 <= 1` on a correct model). A SMELL
    # the human adjudicates (only the domain owner knows if a dip is a bug or stochastic texture) —
    # flag the fingerprint; recommend the property the process actually guarantees.
    for f, tree in trees.items():
        if isinstance(tree, SyntaxError):
            continue
        diff_names = _diff_source_names(tree)
        for node in ast.walk(tree):
            # (1) all()/any() over a per-step sign comparison -> per-step monotonicity claim.
            red, exprs = _reduction(node)
            if red in ("all", "any") and _has_diff_sign_compare(exprs, diff_names):
                violations.append(
                    f"{f.name}:{node.lineno}: per-step monotonicity assertion on np.diff(...) — "
                    f"`{red}(diff <sign> …)` claims the metric improves at (almost) EVERY step, the "
                    f"signature of a BEST-IMPROVING/hill-climbing dynamic. If the update rule is "
                    f"stochastic or non-greedy (random relocation, ε-exploration, annealing) the "
                    f"observable only DRIFTS upward and per-step dips are expected, so this is false "
                    f"against a correct impl. Confirm the dynamic is genuinely monotone; else assert "
                    f"the property it guarantees — net rise with a margin (trend[-1]-trend[0] >= m) "
                    f"plus up-steps OUTNUMBERING down-steps, or a final-vs-baseline distribution "
                    f"test — and smoke-test the gate against a disjoint seed list.")
            # (2) a dip COUNT compared to a CONSTANT threshold: np.sum(diff < -eps) <= 1. The
            #     recommended majority test compares two diff-COUNTS to each other (not a constant),
            #     so requiring a numeric-literal side excludes the good pattern.
            if isinstance(node, ast.Compare) and any(isinstance(op, _ORDER_OPS) for op in node.ops):
                sides = [node.left, *node.comparators]
                counts = [s for s in sides if _is_diff_sign_count(s, diff_names)]
                consts = [s for s in sides if isinstance(s, ast.Constant)
                          and isinstance(s.value, (int, float)) and not isinstance(s.value, bool)]
                if counts and consts:
                    violations.append(
                        f"{f.name}:{node.lineno}: hard dip-count threshold on np.diff(...) — "
                        f"counting per-step sign-violations and comparing to a constant "
                        f"(`np.sum(diff < -eps) <= N`). On an averaged-but-still-stochastic "
                        f"observable the dip COUNT is seed-dependent: a few seeds never smooth a "
                        f"random walk into monotonicity, so the verdict flips with the seed list and "
                        f"the gate measures the seeds, not the model. Replace with a distributional "
                        f"claim the process guarantees — net rise + up-steps outnumber down-steps "
                        f"(count-vs-count, not count-vs-constant), or a final-state-vs-baseline test "
                        f"— and re-run with a disjoint seed list to prove the verdict is stable.")

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
    violations = (lint_spec(project.spec)
                  + lint_frozen_tests(project.code, project.package, project.spec)
                  + lint_dead_modules(project.code, project.package, project.spec)
                  + lint_copied_constants(project.code, project.package))
    if not violations:
        print("[raster lint] frozen-test cross-reference: clean")
        return 0
    print(f"[raster lint] {len(violations)} cross-reference violation(s) "
          f"(each hard-blocks implementation; the green collect can't see them):")
    for v in violations:
        print(f"  - {v}")
    return 1
