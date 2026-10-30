"""Optional CPU autograd identity; preparation never retains model/input state."""

import hashlib
import inspect
import sys
import sysconfig
from functools import cache
from pathlib import Path
from types import FunctionType, ModuleType

import torch

_prepared = None
_completion_guard = None
_completion_factory = None
_capability = None


def register(owner, reader, namespace):
    """Optional version-dependent capability discovery, never an import gate."""
    global _capability
    _capability = None
    try:
        _register(owner, reader, namespace)
    except (AttributeError, TypeError, ValueError):
        pass


def _register(owner, reader, namespace):
    """Freeze original callback definitions at the library definition boundary."""
    global _capability
    if (
        owner.__getattribute__ is not object.__getattribute__
        or inspect.getattr_static(owner, "_readers", None) is not None
        or torch.empty is not torch._C._VariableFunctions.empty
        or namespace["weakref"].ref is not namespace["weakref"].ReferenceType
    ):
        return
    functions, lookups, seen = [], [], set()

    def function(fn):
        if type(fn) is not FunctionType or id(fn) in seen:
            return
        seen.add(id(fn))
        keywords, dictionary = fn.__kwdefaults__, fn.__dict__
        bindings = tuple(
            (n, n in fn.__globals__, fn.__globals__.get(n))
            for n in fn.__code__.co_names
        )
        cells = tuple(c.cell_contents for c in fn.__closure__ or ())
        functions.append(
            (
                fn,
                fn.__code__,
                fn.__defaults__,
                keywords,
                fn.__globals__,
                fn.__closure__,
                dictionary,
                dict(keywords or {}),
                dict(dictionary),
                bindings,
                cells,
            )
        )
        for _, _, value in bindings:
            if type(value) is FunctionType and value.__module__ == fn.__module__:
                function(value)

    def lookup(obj, name):
        value = inspect.getattr_static(obj, name, None)
        lookups.append(
            (obj, name, value, obj.__mro__ if isinstance(obj, type) else None)
        )
        fn = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        function(fn)

    for name in (
        "forward",
        "backward",
        "apply",
        "setup_context",
        "vjp",
        "_backward_cls",
    ):
        lookup(reader, name)
    lookup(reader._backward_cls, "apply")
    lookup(reader._backward_cls, "_forward_cls")
    for base in reader.__mro__[1:]:
        lookup(base, "apply")
    lookup(type(reader), "__getattribute__")
    lookup(owner, "__getattribute__")
    lookup(owner, "_readers")
    for obj, name in (
        (torch, "Tensor"),
        (torch.nn, "Parameter"),
        (torch, "empty"),
        (namespace["weakref"], "ref"),
        (torch._C, "_are_functorch_transforms_active"),
        (torch._C._functorch, "is_functorch_wrapped_tensor"),
        (torch._C, "_get_tracing_state"),
        (torch.cuda, "is_current_stream_capturing"),
    ):
        if type(obj) is not ModuleType:
            return
        lookup(obj, name)
    # A replacement of the class or empty-marker dependencies must also fall
    # back, even when the original class/function objects themselves survive.
    # state is a real module namespace; capture it through its module object.
    module = sys.modules[owner.__module__]
    lookup(module, "_ReadParameter")
    lookup(module, "_identity_saved")
    # Forward AD keeps the original Python Function's jvp behavior. The forced
    # idle value also works if torchcst was imported inside a dual-level scope.
    lookups.append((torch.autograd.forward_ad, "_current_level", -1, None))
    _capability = (
        owner,
        torch.Tensor,
        torch.nn.Parameter,
        tuple(functions),
        tuple(lookups),
    )


@cache
def prepare():
    """Compile once during eager CUDA Algorithm preparation; failure is optional."""
    global _prepared
    _prepared = None
    if (
        _capability is None
        or sys.implementation.name != "cpython"
        or sysconfig.get_config_var("Py_GIL_DISABLED")
    ):
        return None
    try:
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_suffix(".cpp")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        module = load(
            name=f"torchcst_read_lease_{digest}",
            sources=[str(source)],
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
        owner, tensor_type, parameter_type, functions, lookups = _capability
        _prepared = module.make_protector(
            owner,
            tensor_type,
            parameter_type,
            functions,
            lookups,
            torch._C._are_functorch_transforms_active,
            torch._C._functorch.is_functorch_wrapped_tensor,
            torch._C._get_tracing_state,
            torch.cuda.is_current_stream_capturing,
        )
        return _prepared
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return None
