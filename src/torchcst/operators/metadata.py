"""Cached declarations of live Chart/Kernel state, shared by operation bindings."""

import torch

_MODULES = torch.nn.Module.modules
_NAMED_MODULES = torch.nn.Module.named_modules
_GETATTR = torch.nn.Module.__getattr__
_WALK_BEHAVIOR = tuple(
    (fn, fn.__code__, fn.__defaults__, fn.__kwdefaults__)
    for fn in (_MODULES, _NAMED_MODULES, _GETATTR)
)


def _flatten_modules(roots):
    return tuple(module for root in roots for module in root.modules())


def _children(module):
    return tuple((name, id(child)) for name, child in module._modules.items())


def _standard_walk(module):
    """Custom walkers must retain their callbacks on every guard."""
    cls = type(module)
    return (
        cls.modules is _MODULES
        and cls.named_modules is _NAMED_MODULES
        and "modules" not in module.__dict__
        and "named_modules" not in module.__dict__
        and cls.__hash__ is object.__hash__
        and cls.__eq__ is object.__eq__
        and cls.__getattr__ is _GETATTR
        and cls.__getattribute__ is object.__getattribute__
        and all(
            fn.__code__ is code
            and fn.__defaults__ is defaults
            and fn.__kwdefaults__ is keywords
            for fn, code, defaults, keywords in _WALK_BEHAVIOR
        )
        and type(module._modules) is dict
        and type(module._buffers) is dict
    )


class _FixedSignatureCache:
    """One optional native metadata guard and immutable signature.

    Every native check observes live registrations, declaration attributes and
    buffer metadata/versions. Unsupported/custom states use a fresh Python
    scan. Neither path skips a callback boundary. Pickle/deepcopy clear guards.
    """

    def __init__(self):
        self.signature = None
        self.guard = None
        self.guard_api = None
        self.attempted = False
        self.guard_failed = False

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    def current(self, roots):
        if self.guard is not None:
            try:
                if self.guard.check(roots):
                    return self.signature
            except (AttributeError, RuntimeError, TypeError):
                pass
            self.guard_failed = True
        return None


def fixed_state_signature(roots):
    """Check live fixed configuration without reading device values.

    Atom Parameters are excluded. The cache never certifies a later callback:
    every invocation checks registrations and fixed buffer versions again.
    """
    roots = tuple(roots)
    owner = roots[-1] if roots else None
    namespace = getattr(owner, "__dict__", {})
    cached = namespace.get("_fixed_signature_cache")
    if type(cached) is _FixedSignatureCache:
        current = cached.current(roots)
        if current is not None:
            return current
    modules = _flatten_modules(roots)
    markers, buffer_markers = [], []
    cacheable = isinstance(owner, torch.nn.Module)
    for module in modules:
        spec, binding = getattr(module, "spec", None), getattr(module, "binding", None)
        children = _children(module)
        markers.append((id(module), children, id(spec), spec, id(binding), binding))
        # Configuration declarations are metadata. A custom Tensor-valued
        # attribute must not be retained by this metadata cache.
        cacheable = (
            cacheable
            and not isinstance(spec, (torch.Tensor, torch.nn.Module))
            and not isinstance(binding, (torch.Tensor, torch.nn.Module))
        )
        for name, tensor in module._buffers.items():
            if tensor is None:
                continue
            if tensor.is_inference():
                if type(cached) is _FixedSignatureCache:
                    cached.__init__()
                return None
            record = (
                name,
                id(tensor),
                tensor._version,
                tensor.dtype,
                tensor.device,
                tensor.numel(),
                tensor.layout,
            )
            buffer_markers.append((id(module), *record))
    signature = tuple(markers) + tuple(buffer_markers)
    if cacheable:
        cache = (
            cached if type(cached) is _FixedSignatureCache else _FixedSignatureCache()
        )
        same = cache.signature == signature
        if same:
            signature = cache.signature
        namespace["_fixed_signature_cache"] = cache
        from ._native_metadata_guard import guard_api, make_guard

        api = guard_api()
        if (
            not same
            or not cache.attempted
            or cache.guard_api != api
            or cache.guard_failed
        ):
            cache.guard = make_guard(roots, modules, _standard_walk, api)
            cache.guard_api = api
            cache.attempted = True
            cache.guard_failed = False
        cache.signature = signature
    elif type(cached) is _FixedSignatureCache:
        cached.__init__()
    return signature


def operator_declaration(binding):
    """Cache immutable live-operator metadata; refresh configuration outside capture."""
    signature = fixed_state_signature((*binding.cst_charts(), binding.kernel))
    previous = binding.__dict__.get("_execution_declaration")
    if (
        signature is not None
        and previous is not None
        and (previous[0] is signature or previous[0] == signature)
    ):
        return previous[1]
    if (
        binding.execution_parameters().is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "chart/kernel metadata must be refreshed outside CUDA graph capture"
        )
    declaration = binding.declaration()
    binding.__dict__["_execution_declaration"] = (signature, declaration)
    return declaration


# Optional prepared execution may specialize only these original definitions.
from torchcst._builtin_callbacks import seal_functions as _seal_functions

_seal_functions(globals())
