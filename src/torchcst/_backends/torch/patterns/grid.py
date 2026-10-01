"""PyTorch evaluation of grid coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor


def positions(self, indices: Tensor) -> Tensor:
    if indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError("indices must be a one-dimensional long tensor")
    if indices.device != self.start.device:
        raise ValueError("indices and pattern must be on the same device")
    if bool(((indices < 0) | (indices >= self.features)).any()):
        raise IndexError("site index out of bounds")
    remainder = indices
    axes = []
    for size in reversed(self.shape):
        axes.append(remainder % size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    axes.reverse()
    return torch.stack(axes, dim=-1).to(self.start.dtype) * self.spacing + self.start


def bounds(self) -> tuple[Tensor, Tensor]:
    end = self.start + self.spacing * self.start.new_tensor([n - 1 for n in self.shape])
    return self.start, end
