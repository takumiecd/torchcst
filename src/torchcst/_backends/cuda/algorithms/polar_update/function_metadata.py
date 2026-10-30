"""Optional CPython slot checks for the already-certified optimizer callbacks.

Compilation is a cold preparation step. The helper reads function metadata with
its GIL held; it performs no tensor operations or CUDA calls. Unsupported builds
keep the existing PyTorch guard accessors and the identical update algorithm.
"""

import hashlib
import sys
import sysconfig
import weakref
from functools import cache
from pathlib import Path

import torch

_loaded_function_factory = None
_loaded_storage_factory = None
_tensor_data_ptr = torch.Tensor.data_ptr
_tensor_getattribute = torch.Tensor.__getattribute__


@cache
def get_factory():
    global _loaded_function_factory, _loaded_storage_factory
    _loaded_function_factory = _loaded_storage_factory = None
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
        _loaded_function_factory = module.make_guard
        _loaded_storage_factory = module.make_storage_guard
        return _loaded_function_factory
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def prepare_storage_checks(factory, tensors):
    """Batch storage-address checks without retaining Tensor examples.

    Qualification is cold and optional. Unsupported/custom accessors keep the
    existing RootGuard method calls. The native checker detects later accessor
    replacements before calling anything and owns only weak Tensor references.
    """
    if factory is None or factory is not _loaded_function_factory:
        return None
    if any(
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or type(value).__getattribute__ is not _tensor_getattribute
        or type(value).data_ptr is not _tensor_data_ptr
        or "data_ptr" in value.__dict__
        for value in tensors
    ):
        return None
    try:
        checker = _loaded_storage_factory(
            tuple(weakref.ref(value) for value in tensors),
            tuple(value.data_ptr for value in tensors),
        )
        return checker if checker(None) else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
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
