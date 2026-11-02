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


class Placement:
    """Device/dtype hints for tensor-free declarations before atoms are bound."""

    def __init__(self, *, device=None, dtype=None):
        self.device = torch.device("cpu" if device is None else device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = _floating_dtype(dtype)

    def converted(self, fn):
        # Module.to() supplies a Tensor conversion callback. Probe placement with
        # an empty temporary; retain neither Tensor nor allocator storage.
        value = fn(torch.empty(0, device=self.device, dtype=self.dtype))
        return Placement(device=value.device, dtype=value.dtype)
