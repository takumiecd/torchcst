"""PyTorch evaluation of lazy coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry.geometry import (
    EuclideanGeometry,
)


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


def positions(self, indices: Tensor) -> Tensor:
    raise NotImplementedError


def _embed(self, coordinates: Tensor) -> Tensor:
    return self.geometry.lift_chart_coordinates(coordinates)


def squared_distance(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    sites = self.positions(self._indices(selection))
    if not isinstance(self.geometry, EuclideanGeometry):
        self.geometry.validate_points(sites, name="sites")
    return self.geometry.squared_distance(sites, centers)


def center_offsets(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    sites = self.positions(self._indices(selection))
    if not isinstance(self.geometry, EuclideanGeometry):
        self.geometry.validate_points(sites, name="sites")
    return self.geometry.center_offsets(sites, centers)


def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    if mode not in ("balanced", "uniform"):
        raise ValueError("mode must be 'balanced' or 'uniform'")
    if isinstance(self.geometry, EuclideanGeometry) and mode == "uniform":
        low, high = self._bounds()
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
        return self.geometry.initialize_centers(
            self.positions(indices), atoms, mode="balanced"
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
    return self.geometry.initialize_centers(self.positions(indices), atoms, mode=mode)


def _unravel(indices: Tensor, shape: tuple[int, ...]) -> tuple[Tensor, ...]:
    remainder = indices
    coordinates = []
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    return tuple(reversed(coordinates))
