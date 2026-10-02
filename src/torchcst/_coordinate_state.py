"""Shared configuration-boundary helpers for coordinate Tensor owners."""

import torch


def _coordinates(tensor):
    return tuple(tuple(row) for row in tensor.detach().cpu().tolist())


def _floating_dtype(dtype):
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("coordinate state requires a floating-point dtype")
    return dtype


def _validate_loaded_state(state, incompatible_keys):
    # Loading is an explicit boundary; validate current buffers rather than the
    # constructor snapshot. These reads never occur in the training hot path.
    try:
        state.declaration()
    except (ValueError, TypeError) as error:
        raise RuntimeError("invalid coordinate checkpoint configuration") from error
