"""PyTorch evaluation of product coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.charts.lazy import _unravel
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.patterns import execution as _patterns


def positions(self, indices: Tensor) -> Tensor:
    if (
        indices.dtype != torch.long
        or indices.ndim != 1
        or indices.device != self.device
    ):
        raise ValueError(
            "indices must be a one-dimensional long tensor on the chart device"
        )
    if bool(((indices < 0) | (indices >= self.features)).any()):
        raise IndexError("site index out of bounds")
    return _geometry.lift_chart_coordinates(
        self.geometry,
        torch.cat(
            tuple(
                _patterns.positions(axis, axis_index)
                for axis, axis_index in zip(self.axes, _unravel(indices, self.shape))
            ),
            dim=-1,
        ),
    )


def _bounds(self) -> tuple[Tensor, Tensor]:
    bounds = [_patterns.bounds(axis) for axis in self.axes]
    return torch.cat(tuple(low for low, _ in bounds)), torch.cat(
        tuple(high for _, high in bounds)
    )
