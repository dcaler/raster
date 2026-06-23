"""Fallback-only import stub for the not-yet-built product package — a pytest plugin
raster injects (`-p raster._freezestub`) ONLY for freeze-phase collects.

A frozen test legitimately does `from <product>.x import Y` for a package that does not
exist yet at freeze time. `pytest --collect-only` *imports* each test module, so without
this it dies on ModuleNotFoundError and the worker is handed a repair instruction it
cannot satisfy (the import is required; no edit fixes it) — a silent, think-burning loop.

This installs a meta-path finder that resolves the product package to a permissive dummy
so collection proceeds, WITHOUT ever masking a real broken import:

  * Fallback-only — appended to the END of sys.meta_path (lowest precedence), and it
    no-ops entirely if the real product package is already importable. The moment real
    code exists it always wins; the stub never fires.
  * Phase-scoped — raster sets RASTER_STUB_PACKAGE and loads this plugin only for
    freeze-phase collects (P0.* authoring + a `--collect-only` freeze gate), never for an
    implementation gate, so it cannot make a post-impl gate pass vacuously.

Collection imports test modules but does not run test bodies, so a permissive dummy is
enough: `from pkg.sub import Name` resolves and module-level decorators referencing it
won't explode.
"""

import importlib.util
import os
import sys
import types
from importlib.machinery import ModuleSpec


class _Dummy:
    """A permissive value: callable, attributable, iterable and subscriptable, so any use
    a frozen test makes of a stubbed name during *collection* stays inert."""
    def __call__(self, *a, **k):
        return _Dummy()

    def __getattr__(self, name):
        return _Dummy()

    def __iter__(self):
        return iter(())

    def __getitem__(self, key):
        return _Dummy()


class _DummyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)   # let import machinery see real dunders as absent
        return _Dummy()


class _StubFinder:
    """Resolve the product root and any submodule to a dummy package module."""
    def __init__(self, root: str):
        self.root = root
        self.prefix = root + "."

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.root or fullname.startswith(self.prefix):
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        mod = _DummyModule(spec.name)
        mod.__path__ = []                # mark as a package so submodules keep resolving
        return mod

    def exec_module(self, module):
        pass


def _install() -> None:
    root = os.environ.get("RASTER_STUB_PACKAGE", "").strip()
    if not root:
        return
    try:                                 # real package already importable -> never stub it
        if importlib.util.find_spec(root) is not None:
            return
    except Exception:                    # a broken real package: don't mask it either
        return
    if not any(isinstance(f, _StubFinder) and f.root == root for f in sys.meta_path):
        sys.meta_path.append(_StubFinder(root))   # APPEND => lowest precedence (fallback-only)


_install()
