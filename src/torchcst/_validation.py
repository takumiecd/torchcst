"""Shared argument-validation helpers.

Private to torchcst. Every helper raises with the caller-supplied ``name`` so
error messages stay as specific as the hand-written checks they replace.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as an int, rejecting bools and values below ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        if minimum == 1:
            raise ValueError(f"{name} must be positive")
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def as_id_vector(value: object, name: str) -> Tensor:
    """Return a detached CPU clone of a rank-1 int64 entity-id tensor."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be rank 1")
    if value.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype int64")
    return value.detach().to(device="cpu").clone()


def cat_or_empty(tensors: Sequence[Tensor], *, like: Tensor | None = None) -> Tensor:
    """Concatenate ``tensors``, or return the right empty tensor for none.

    With ``like`` the empty case matches that tensor's dtype/device (detached);
    without it the empty case is the CPU int64 vector used for entity ids.
    """
    if tensors:
        return torch.cat(list(tensors))
    if like is not None:
        return like.detach().new_zeros((0,))
    return torch.zeros(0, dtype=torch.int64)
