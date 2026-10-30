"""Optional CPython slot checks for the already-certified optimizer callbacks.

Compilation is a cold preparation step. The helper reads function metadata with
its GIL held; it performs no tensor operations or CUDA calls. Unsupported builds
keep the existing PyTorch guard accessors and the identical update algorithm.
"""

import hashlib
import sys
import sysconfig
from functools import cache
from pathlib import Path

# Group guards only borrow an already-loaded helper. They never start a build
# after group validation: compilation/import can invoke application callbacks.
_loaded_factory = None


@cache
def get_factory():
    global _loaded_factory
    _loaded_factory = None
    if sys.implementation.name != "cpython" or sysconfig.get_config_var(
        "Py_GIL_DISABLED"
    ):
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        module = load(
            name=f"torchcst_polar_host_guard_{digest}",
            sources=[str(source)],
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
        _loaded_factory = module.make_guard
        return _loaded_factory
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def prepare_checks(factory, functions, owners):
    """Return a slot checker and unique dictionaries whose versions stay guarded.

    The slot checker also checks dictionary identity. The existing RootGuard
    then reads each dictionary directly, avoiding function attribute traversal.
    Dictionaries already checked through their owner need no duplicate node.
    """
    if factory is None:
        return None, ()
    try:
        checker = factory(functions)
        checked = {
            id(owner if type(owner) is dict else owner.__dict__)
            for owner in owners
            if not isinstance(owner, type)
        }
        dictionaries = []
        for function in functions:
            for value in (
                function.__dict__,
                function.__globals__,
                function.__kwdefaults__,
            ):
                if value is not None and id(value) not in checked:
                    checked.add(id(value))
                    dictionaries.append(value)
        if not checker(None):
            return None, ()
        return checker, tuple(dictionaries)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None, ()
