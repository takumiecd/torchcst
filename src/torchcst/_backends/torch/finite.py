"""Batch finite-value checks without changing tensors or optimizer arithmetic."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_enabled = ContextVar("torchcst_finite_checks", default=True)


def checks_enabled():
    return _enabled.get()


@contextmanager
def checking(enabled):
    """Scope optional value checks to one optimizer invocation/thread.

    Public geometry calls outside that invocation retain their validation.
    Structural checks and fixed configuration validation are unconditional.
    """
    if type(enabled) is not bool:
        raise TypeError("check_finite must be a bool")
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


@torch.no_grad()
def first_nonfinite(tensors):
    """Return the first invalid tensor index, or None when all are finite.

    CUDA real floating tensors share one host transfer per device/dtype group.
    The infinity norm only takes an absolute maximum: unlike a squared norm it
    cannot overflow for finite real inputs. Complex tensors use isfinite
    directly because their magnitude can overflow with finite components.
    Empty tensors are finite. CPU and other layouts keep the reference check.
    No activation-sized mask, concatenation or persistent cache is created.
    """
    if not checks_enabled():
        return None
    groups = {}
    invalid = []
    real_types = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    for index, value in enumerate(tensors):
        if not value.numel():
            continue
        if (
            value.is_cuda
            and value.layout == torch.strided
            and value.dtype in real_types
        ):
            groups.setdefault((value.device, value.dtype), []).append((index, value))
        elif not bool(torch.isfinite(value).all()):
            invalid.append(index)

    for entries in groups.values():
        norms = torch._foreach_norm([value for _, value in entries], float("inf"))
        # One small transfer for the group. tolist on this CUDA vector waits
        # once; iterating over individual CUDA scalar tensors would wait N times.
        valid = torch.isfinite(torch.stack(norms)).tolist()
        invalid.extend(index for (index, _), ok in zip(entries, valid) if not ok)
    return min(invalid) if invalid else None
