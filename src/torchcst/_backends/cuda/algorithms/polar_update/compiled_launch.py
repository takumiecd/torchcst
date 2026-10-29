"""Replay the identical compiled Triton kernel without JIT argument binding.

The first ordinary Algorithm execution supplies this handle. This module never
compiles a different equation or retains previous/proposal invocation tensors.
Unsupported runtimes, hooks and changed specializations use Algorithm.execute.
"""

import __future__

import sys
from types import FunctionType
from typing import Generic

import torch

_REVERSED = reversed


def _hook_loop(self, *args, **kwargs):
    # Reference bytecode for Triton's standard HookChain, never executed here.
    for call in self.calls if not self.reversed else reversed(self.calls):
        call(*args, **kwargs)


def _empty_hook_certificate(knobs):
    """Qualify the standard loop; unknown/custom implementations stay active."""
    cls = getattr(knobs, "HookChain", None)
    if type(cls) is not type or cls.__mro__ not in (
        (cls, object),
        (cls, Generic, object),
    ):
        return None
    fn = cls.__dict__.get("__call__")
    if type(fn) is not FunctionType:
        return None
    expected, actual = _hook_loop.__code__, fn.__code__
    fields = (
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_argcount",
        "co_kwonlyargcount",
        "co_freevars",
        "co_cellvars",
        "co_exceptiontable",
    )
    if (
        cls.__getattribute__ is not object.__getattribute__
        or any(
            name in base.__dict__
            for base in cls.__mro__
            for name in ("calls", "reversed", "__getattr__")
        )
        or any(
            getattr(actual, name, None) != getattr(expected, name, None)
            for name in fields
        )
        or (actual.co_flags ^ expected.co_flags) & ~__future__.annotations.compiler_flag
        or fn.__defaults__ is not None
        or fn.__kwdefaults__ is not None
    ):
        return None
    return cls, fn, actual


_SCALARS = (
    "amplitude_max",
    "w_c",
    "activity_gain",
    "dormant_expansion_rate",
    "radial_regularization",
)
_ARGUMENTS = (
    "OLD",
    "PROPOSAL",
    "AMAX",
    "WC",
    "GAIN",
    "DORMANT",
    "RADIAL",
    "COUNT",
    "STEP",
    "TIME_ENERGY",
    "BLOCK",
    "AMPLITUDE_DECAY",
    "HAS_AMPLITUDE_DECAY",
    "INPUT_PERIOD",
    "OUTPUT_PERIOD",
    "PERIODIC",
)


class CompiledPolarLaunch:
    def __init__(self, binding, record, knobs, device):
        kernel, self.jit, self.count, self.step, self.time_energy, self.amount = record
        self.kernel = kernel
        self.key = self.jit.cache_key
        self.knobs = knobs
        self.empty_hook = _empty_hook_certificate(knobs)
        self.device = device
        self.debug = (
            self.jit.debug,
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
        )
        self.scalars = tuple(binding.kernel.scalar(name) for name in _SCALARS)
        self.periods = tuple(chart.geometry.periods for chart in binding.cst_charts())
        self.runner = kernel[((self.count + 127) // 128, 1, 1)]
        module = sys.modules[self.jit.fn.__module__]
        # Dictionary guards also preserve JIT's used-global checks and explicit
        # cache invalidation. They do not read any device scalar values.
        dictionaries = [self.jit.device_caches]
        cache = self.jit.device_caches[device]
        self.cache_entry = cache
        dictionaries.append(cache[0])
        dictionaries.append(self.jit.used_global_vals)
        dictionaries.extend(entry[1] for entry in self.jit.used_global_vals.values())
        seen = set()
        self.guarded_owners = [
            (module, ()),
            (self.jit, ()),
            (type(self.jit), ()),
            (kernel, ()),
            (type(kernel), ()),
        ]
        if self.empty_hook is not None:
            self.guarded_owners.append((self.empty_hook[0], ()))
        for dictionary in dictionaries:
            if type(dictionary) is dict and id(dictionary) not in seen:
                seen.add(id(dictionary))
                self.guarded_owners.append((dictionary, ()))

    def enabled(self):
        """Pre-run hooks keep their JIT path; launch hooks use the same runner."""
        runtime = self.knobs.runtime
        hooks = self.jit.pre_run_hooks
        mode = self.knobs.compilation.instrumentation_mode
        return (
            type(hooks) is list
            and len(hooks) == 0
            and self.jit.cache_key == self.key
            and type(runtime.debug) is bool
            and runtime.debug is self.debug[1]
            and self.jit.debug is self.debug[0]
            and type(mode) is type(self.debug[2])
            and mode == self.debug[2]
            and self.jit.device_caches.get(self.device) is self.cache_entry
            and torch.cuda.current_device() == self.device
        )

    def hooks_active(self):
        """Empty standard HookChains execute no user code and need no recheck."""
        runtime = self.knobs.runtime
        for hook in (runtime.launch_enter_hook, runtime.launch_exit_hook):
            if hook is None:
                continue
            if self.empty_hook is None:
                return True
            cls, fn, code = self.empty_hook
            if (
                type(hook) is not cls
                or cls.__call__ is not fn
                or fn.__code__ is not code
                or fn.__defaults__ is not None
                or fn.__kwdefaults__ is not None
            ):
                return True
            values = hook.__dict__
            calls = values.get("calls")
            reverse = values.get("reversed")
            if type(calls) is not list or calls or type(reverse) is not bool:
                return True
            if reverse and (
                fn.__globals__.get("reversed", _REVERSED) is not _REVERSED
                or fn.__builtins__.get("reversed") is not _REVERSED
            ):
                return True
        return False

    def run(self, previous, point, inputs):
        """Return False without writes if a new specialization is required."""
        amount = inputs.step_size * inputs.amplitude_weight_decay
        if (
            type(inputs.step_size) is not type(self.step)
            or inputs.step_size != self.step
            or type(amount) is not type(self.amount)
            or amount != self.amount
            # The captured specialization used ordinary aligned CUDA tensors.
            or previous.data_ptr() % 16
        ):
            return False
        arguments = (
            previous,
            point,
            *self.scalars,
            self.count,
            self.step,
            self.time_energy,
            128,
            amount,
            inputs.amplitude_weight_decay != 0,
            *self.periods,
            True,
        )
        # Tensor objects, not cached integer pointers, are passed to Triton's
        # existing launcher. Scalars remain live on device, as in the JIT path.
        if any(value.data_ptr() % 16 for value in (*self.scalars, *self.periods)):
            return False
        self.runner(*arguments)
        torch.autograd.graph.increment_version(point)
        return True


def prepare_launch(binding, state):
    record = getattr(state, "_polar_launch", None)
    if type(record) is not tuple or len(record) != 6 or record[0] is None:
        return None
    try:
        from triton import knobs

        kernel, jit = record[:2]
        if (
            tuple(jit.arg_names) != _ARGUMENTS
            or kernel.src.fn is not jit
            or not callable(kernel.__getitem__)
            or any(binding.kernel.scalar(name).data_ptr() % 16 for name in _SCALARS)
        ):
            return None
        result = CompiledPolarLaunch(
            binding, record, knobs, torch.cuda.current_device()
        )
        return result if result.enabled() else None
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError):
        return None
