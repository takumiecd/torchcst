"""PyTorch evaluation of lazy coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst.geometry.spec import EuclideanGeometrySpec


def _indices(self, selection: slice | Tensor | None) -> Tensor:
    if selection is None:
        return torch.arange(self.features, device=self.device)
    if isinstance(selection, slice):
        start, stop, step = selection.indices(self.features)
        return torch.arange(start, stop, step, device=self.device)
    if (
        not isinstance(selection, Tensor)
        or selection.dtype != torch.long
        or selection.ndim != 1
    ):
        raise TypeError("selection must be a slice or one-dimensional long tensor")
    return selection


def squared_distance(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    sites = _charts.positions(self, _indices(self, selection))
    if not type(self.geometry.spec) is EuclideanGeometrySpec:
        _geometry.validate_points(self.geometry, sites, name="sites")
    return _geometry.squared_distance(self.geometry, sites, centers)


def center_offsets(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    sites = _charts.positions(self, _indices(self, selection))
    if not type(self.geometry.spec) is EuclideanGeometrySpec:
        _geometry.validate_points(self.geometry, sites, name="sites")
    return _geometry.center_offsets(self.geometry, sites, centers)


def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    if type(self.geometry.spec) is EuclideanGeometrySpec and mode == "uniform":
        low, high = _charts._bounds(self)
        return low + torch.rand(
            atoms, self.embedding_dim, device=self.device, dtype=self.dtype
        ) * (high - low)
    if mode == "balanced":
        indices = (
            torch.linspace(
                0,
                self.features - 1,
                atoms,
                device=self.device,
                dtype=torch.float64,
            )
            .round()
            .long()
        )
        return _geometry.initialize_centers(
            self.geometry, _charts.positions(self, indices), atoms, mode="balanced"
        )
    # Other geometries receive a bounded representative site sample.
    sample_count = min(self.features, max(atoms, 1024))
    indices = (
        torch.linspace(
            0,
            self.features - 1,
            sample_count,
            device=self.device,
            dtype=torch.float64,
        )
        .round()
        .long()
    )
    return _geometry.initialize_centers(
        self.geometry, _charts.positions(self, indices), atoms, mode=mode
    )


def _unravel(indices: Tensor, shape: tuple[int, ...]) -> tuple[Tensor, ...]:
    remainder = indices
    coordinates = []
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    return tuple(reversed(coordinates))
