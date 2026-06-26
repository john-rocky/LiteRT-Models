"""Shared scipy/_propack + getsourcefile stub for macOS probes.

Import this FIRST (before transformers / litert_torch) in any probe script:
    import _stub  # noqa: F401
Mirrors the guards baked into probe_convert.py so raw-module probes can run too.
"""
import sys, types, inspect


class _D:
    def __getattr__(self, n):
        return lambda *a, **k: None

    def __call__(self, *a, **k):
        return None


_pp = types.ModuleType("scipy.sparse.linalg._propack")
_pp.__file__ = "<stub:scipy._propack>"
_pp.__spec__ = None
for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _D())
sys.modules["scipy.sparse.linalg._propack"] = _pp

_opt = types.ModuleType("scipy.optimize")
_opt.__file__ = "<stub:scipy.optimize>"
_opt.__spec__ = None
_opt.linear_sum_assignment = lambda *a, **k: None
sys.modules["scipy.optimize"] = _opt

_orig_gsf = inspect.getsourcefile


def _safe_getsourcefile(obj):
    try:
        return _orig_gsf(obj)
    except (AttributeError, TypeError):
        nm = getattr(obj, "__name__", repr(obj))
        sys.stderr.write(f"[probe] guarded getsourcefile crash on module {nm!r}\n")
        return None


inspect.getsourcefile = _safe_getsourcefile
