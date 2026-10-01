"""PyTorch evaluation of strip coordinate declarations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst._backends.torch.charts.lazy import _unravel
from torchcst.geometry.geometry import (
    EuclideanGeometry,
    SphereGeometry,
    TorusGeometry,
)


def tile_indices(self, station: int) -> tuple[Tensor, Tensor]:
    """Return logical and local flat indices for one physical tile."""

    if type(station) is not int or not 0 <= station < self.tile_count:
        raise IndexError("tile station out of bounds")
    local_axes = torch.meshgrid(
        *(torch.arange(size, device=self.device) for size in self.tile_shape),
        indexing="ij",
    )
    valid = torch.ones(self.tile_shape, dtype=torch.bool, device=self.device)
    logical = torch.zeros(self.tile_shape, dtype=torch.long, device=self.device)
    local = torch.zeros_like(logical)
    logical_stride = 1
    local_stride = 1
    for index in reversed(range(len(self.shape))):
        coordinate = local_axes[index]
        global_coordinate = (
            coordinate + station * self.tile_shape[index]
            if index == self.axis
            else coordinate
        )
        valid &= global_coordinate < self.shape[index]
        logical += global_coordinate * logical_stride
        local += coordinate * local_stride
        logical_stride *= self.shape[index]
        local_stride *= self.tile_shape[index]
    return logical[valid], local[valid]


def validate_support(self, radius: float) -> None:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("kernel support radius must be positive")
    if self.tile_count <= 2:
        return
    if isinstance(self.geometry, TorusGeometry):
        self._validate_torus_support(radius)
        return
    if not isinstance(self.geometry, (EuclideanGeometry, SphereGeometry)):
        raise NotImplementedError(
            "strip support bounds require Euclidean or Sphere geometry"
        )
    line = self.axes[self.axis]
    span = float(line.spacing[0]) * (self.tile_shape[self.axis] - 1)
    nonneighbor_gap = 2 * float(self.tile_pitch) - span
    if isinstance(self.geometry, SphereGeometry):
        low, high = self._bounds()
        bound = torch.maximum(low.abs(), high.abs())
        max_norm_squared = float(bound.square().sum())
        sphere_radius = float(self.geometry.radius)
        scale_floor = sphere_radius**3 / (sphere_radius**2 + max_norm_squared) ** 1.5
        nonneighbor_gap *= scale_floor
    if nonneighbor_gap <= 2 * radius:
        raise ValueError("strip support radius reaches more than two tile stations")


def _validate_torus_support(self, radius: float) -> None:
    """Check second-neighbor stations, the nearest nonadjacent pairs."""

    geometry = self.geometry
    assert isinstance(geometry, TorusGeometry)
    line = self.axes[self.axis]
    start = float(line.start[0])
    spacing = float(line.spacing[0])
    pitch = float(self.tile_pitch)
    period = geometry.circumference
    intervals = []
    for station in range(self.tile_count):
        count = min(
            self.tile_shape[self.axis],
            self.shape[self.axis] - station * self.tile_shape[self.axis],
        )
        low = start + station * pitch
        intervals.append((low, low + (count - 1) * spacing))
    # Ordered, disjoint intervals make cyclic second-neighbor gaps the
    # smallest candidates among all nonadjacent station pairs.
    for station in range(self.tile_count):
        left, right = sorted((station, (station + 2) % self.tile_count))
        direct_gap = intervals[right][0] - intervals[left][1]
        wrapped_gap = period - (intervals[right][1] - intervals[left][0])
        gap = min(direct_gap, wrapped_gap)
        if geometry.axis_separation_lower_bound(gap) <= 2 * radius:
            raise ValueError("strip support radius reaches more than two tile stations")


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
    coordinates = []
    for index, (pattern, axis_index) in enumerate(
        zip(self.axes, _unravel(indices, self.shape))
    ):
        if index == self.axis:
            station = torch.div(
                axis_index, self.tile_shape[index], rounding_mode="floor"
            )
            local_index = axis_index % self.tile_shape[index]
            coordinates.append(
                pattern.positions(local_index)
                + station[:, None].to(self.dtype) * self.tile_pitch
            )
        else:
            coordinates.append(pattern.positions(axis_index))
    return self._embed(torch.cat(tuple(coordinates), dim=-1))


def _bounds(self) -> tuple[Tensor, Tensor]:
    lows = []
    highs = []
    for index, pattern in enumerate(self.axes):
        if index == self.axis:
            start = pattern.positions(
                torch.zeros(1, dtype=torch.long, device=self.device)
            )[0]
            last_local = (self.shape[index] - 1) % self.tile_shape[index]
            end = (
                pattern.positions(
                    torch.tensor([last_local], dtype=torch.long, device=self.device)
                )[0]
                + (self.tile_count - 1) * self.tile_pitch
            )
            lows.append(start)
            highs.append(end)
        else:
            low, high = pattern.bounds()
            lows.append(low)
            highs.append(high)
    return torch.cat(tuple(lows)), torch.cat(tuple(highs))
