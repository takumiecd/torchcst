"""Reuse only the registration of CST's final AdamW pre-hook.

The invocation callback is cleared after every step. The dormant registration
holds no optimizer, model, snapshot or gradient; collection removes it too.
"""

import weakref
from collections import OrderedDict

from torch.optim import Optimizer

_REGISTER = Optimizer.register_step_pre_hook
_REGISTER_CODE = _REGISTER.__code__
_REGISTER_DEFAULTS = _REGISTER.__defaults__
_REGISTER_KEYWORDS = _REGISTER.__kwdefaults__
_HOOK_MODULE = _REGISTER.__globals__["hooks"]
_HANDLE = _HOOK_MODULE.RemovableHandle
_HANDLE_METHODS = tuple(
    (name, getattr(_HANDLE, name), getattr(_HANDLE, name).__code__)
    for name in ("__init__", "remove")
)


class _CachedHook:
    def __init__(self):
        self.callback = self.handle = self.table = self.finalizer = None

    def __call__(self, base, args, kwargs):
        if self.callback is not None:
            return self.callback(base, args, kwargs)
        return None

    def close(self):
        self.callback = None
        if self.handle is not None:
            self.handle.remove()
        if self.finalizer is not None:
            self.finalizer.detach()
        self.handle = self.table = self.finalizer = None


def clear(owner):
    cached = owner.__dict__.pop("_amplitude_pre_hook", None)
    if cached is not None:
        cached.close()


def install(owner, callback, *, reuse):
    base = owner.base_optimizer
    cached = owner.__dict__.get("_amplitude_pre_hook")
    if cached is not None and cached.callback is not None:
        raise RuntimeError("reentrant CST amplitude-decay steps are unsupported")
    if not reuse:
        clear(owner)
        return base.register_step_pre_hook(callback), False
    table = base._optimizer_step_pre_hooks
    standard = (
        type(table) is OrderedDict
        and "register_step_pre_hook" not in base.__dict__
        and type(base).register_step_pre_hook is _REGISTER
        and _REGISTER.__code__ is _REGISTER_CODE
        and _REGISTER.__defaults__ is _REGISTER_DEFAULTS
        and _REGISTER.__kwdefaults__ is _REGISTER_KEYWORDS
        and _REGISTER.__globals__.get("hooks") is _HOOK_MODULE
        and _HOOK_MODULE.RemovableHandle is _HANDLE
        and all(
            getattr(_HANDLE, name) is fn and fn.__code__ is code
            for name, fn, code in _HANDLE_METHODS
        )
    )
    if not standard:
        clear(owner)
        return base.register_step_pre_hook(callback), False
    if cached is None:
        cached = _CachedHook()
        owner.__dict__["_amplitude_pre_hook"] = cached
    if cached.table is not table or table.get(cached.handle.id) is not cached:
        # A replaced/copied hook table may still contain the old callable.
        for key in [key for key, value in table.items() if value is cached]:
            del table[key]
        cached.close()
        cached.handle = base.register_step_pre_hook(cached)
        cached.table = table
        cached.finalizer = weakref.finalize(owner, cached.handle.remove)
    # Users can register more local hooks between steps. CST remains LAST,
    # exactly where a newly registered temporary callback would have run.
    table.move_to_end(cached.handle.id)
    cached.callback = callback
    return cached, True


def finish(owner, handle, reusable, *, success):
    if reusable:
        handle.callback = None
        if not success:
            clear(owner)
    else:
        handle.remove()
