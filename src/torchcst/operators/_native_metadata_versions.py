"""Optional batched buffer-version epilogue; importing this never compiles."""

import hashlib
import sys
import sysconfig
from functools import cache
from pathlib import Path

_factory = None


@cache
def prepare():
    """Called at an execution preparation boundary, with a Python fallback."""
    global _factory
    _factory = None
    if sys.implementation.name != "cpython" or sysconfig.get_config_var(
        "Py_GIL_DISABLED"
    ):
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        module = load(
            name="torchcst_fixed_versions_" + digest,
            sources=[str(source)],
            with_cuda=False,
            extra_cflags=["-O3"],
            verbose=False,
        )
        _factory = module.make_guard
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return None
    return _factory


def refresh(binding):
    """Rebuild an earlier fixed guard once when the optional factory appears."""
    factory = prepare()
    if factory is None:
        return
    from .metadata import _FixedSignatureCache

    cache = getattr(binding.kernel, "__dict__", {}).get("_fixed_signature_cache")
    if (
        type(cache) is _FixedSignatureCache
        and getattr(cache, "_version_factory", None) is not factory
    ):
        cache.guard = None
        cache.attempted = False
        cache._version_factory = factory
