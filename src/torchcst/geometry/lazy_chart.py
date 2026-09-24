"""Logical operator charts with index-based, bounded coordinate generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
import torch
from torch import Tensor, nn

from .chart import Chart
from .geometry import EuclideanGeometry, Geometry, SphereGeometry
from .pattern import LinePattern, SitePattern


class _LazyChart(Chart):
    """A Chart whose sites are computed only for requested flat indices."""

    def __init__(self, shape: tuple[int, ...], geometry: Geometry) -> None:
        nn.Module.__init__(self)
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a Geometry")
        self._shape = shape
        self.geometry = geometry

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def features(self) -> int:
        return math.prod(self.shape)

    @property
    def dim(self) -> int:
        return self.embedding_dim

    @property
    def device(self) -> torch.device:
        return self.reference.device

    @property
    def dtype(self) -> torch.dtype:
        return self.reference.dtype

    @property
    def trainable(self) -> bool:
        return False

    @property
    def spacing(self) -> None:
        return None

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
        if isinstance(self.geometry, SphereGeometry):
            return self.geometry.lift_tangent_sites(coordinates)
        return coordinates

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
        return self.geometry.initialize_centers(
            self.positions(indices), atoms, mode=mode
        )

    def get_extra_state(self) -> dict[str, object]:
        return {
            **super().get_extra_state(),
            "shape": self.shape,
            "layout": self._layout(),
        }

    def _layout(self) -> tuple:
        raise NotImplementedError


def _validate_axes(
    shape: Sequence[int], axes: Sequence[SitePattern]
) -> tuple[tuple[int, ...], tuple[SitePattern, ...]]:
    shape = tuple(shape)
    axes = tuple(axes)
    if not shape or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError("shape must contain positive integers")
    if len(axes) != len(shape):
        raise ValueError("axes must contain one SitePattern per shape dimension")
    if any(not isinstance(axis, SitePattern) for axis in axes):
        raise TypeError("every axis must be a SitePattern")
    if any(axis.features != size for axis, size in zip(axes, shape)):
        raise ValueError("each axis pattern size must match its shape dimension")
    reference = axes[0].reference
    if any(
        axis.reference.device != reference.device
        or axis.reference.dtype != reference.dtype
        for axis in axes[1:]
    ):
        raise ValueError("axis patterns must share device and dtype")
    return shape, axes


def _unravel(indices: Tensor, shape: tuple[int, ...]) -> tuple[Tensor, ...]:
    remainder = indices
    coordinates = []
    for size in reversed(shape):
        coordinates.append(remainder % size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    return tuple(reversed(coordinates))


class ProductChart(_LazyChart):
    """Lazy Cartesian layout of one site pattern per logical tensor axis."""

    def __init__(
        self,
        shape: Sequence[int],
        axes: Sequence[SitePattern],
        *,
        geometry: Geometry | None = None,
    ) -> None:
        shape, axes = _validate_axes(shape, axes)
        dim = sum(axis.dim for axis in axes)
        super().__init__(shape, geometry or EuclideanGeometry(dim))
        if (
            isinstance(self.geometry, SphereGeometry)
            and self.geometry.intrinsic_dim != dim
        ) or (
            not isinstance(self.geometry, SphereGeometry) and self.embedding_dim != dim
        ):
            raise ValueError(
                "geometry dimensions must match combined axis pattern dimensions"
            )
        self.axes = nn.ModuleList(axes)

    @property
    def reference(self) -> Tensor:
        return self.axes[0].reference

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
        return self._embed(
            torch.cat(
                tuple(
                    axis.positions(axis_index)
                    for axis, axis_index in zip(
                        self.axes, _unravel(indices, self.shape)
                    )
                ),
                dim=-1,
            )
        )

    def _bounds(self) -> tuple[Tensor, Tensor]:
        bounds = [axis.bounds() for axis in self.axes]
        return torch.cat(tuple(low for low, _ in bounds)), torch.cat(
            tuple(high for _, high in bounds)
        )

    def _layout(self) -> tuple:
        return tuple(type(axis).__qualname__ for axis in self.axes)


class StripChart(_LazyChart):
    """Tile one LinePattern axis while retaining the product geometry dimension."""

    def __init__(
        self,
        shape: Sequence[int],
        tile_shape: Sequence[int],
        *,
        axes: Sequence[SitePattern],
        axis: int,
        tile_pitch: float,
        geometry: Geometry | None = None,
    ) -> None:
        shape, axes = _validate_axes(shape, axes)
        tile_shape = tuple(tile_shape)
        if len(tile_shape) != len(shape) or any(
            type(size) is not int or size < 1 or size > full
            for size, full in zip(tile_shape, shape)
        ):
            raise ValueError("tile_shape must match shape and fit every axis")
        if type(axis) is not int or not 0 <= axis < len(shape):
            raise ValueError("axis must select a shape dimension")
        if not isinstance(axes[axis], LinePattern):
            raise TypeError("the strip axis must use a one-dimensional LinePattern")
        if any(
            tile != full
            for index, (tile, full) in enumerate(zip(tile_shape, shape))
            if index != axis
        ):
            raise ValueError("only the selected LinePattern axis may be tiled")
        if not math.isfinite(tile_pitch) or tile_pitch <= 0:
            raise ValueError("tile_pitch must be positive")
        line = axes[axis]
        local_span = float(line.spacing[0]) * (tile_shape[axis] - 1)
        if math.ceil(shape[axis] / tile_shape[axis]) > 1 and tile_pitch <= local_span:
            raise ValueError("tile_pitch must exceed the width of one tile")
        dim = sum(pattern.dim for pattern in axes)
        super().__init__(shape, geometry or EuclideanGeometry(dim))
        if (
            isinstance(self.geometry, SphereGeometry)
            and self.geometry.intrinsic_dim != dim
        ) or (
            not isinstance(self.geometry, SphereGeometry) and self.embedding_dim != dim
        ):
            raise ValueError(
                "geometry dimensions must match combined axis pattern dimensions"
            )
        self.axes = nn.ModuleList(axes)
        self.axis = axis
        self.tile_shape = tile_shape
        self.register_buffer("tile_pitch", line.reference.new_tensor(float(tile_pitch)))

    @property
    def reference(self) -> Tensor:
        return self.tile_pitch

    @property
    def tile_grid(self) -> tuple[int, ...]:
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_shape))

    @property
    def tile_count(self) -> int:
        return math.prod(self.tile_grid)

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
            scale_floor = (
                sphere_radius**3 / (sphere_radius**2 + max_norm_squared) ** 1.5
            )
            nonneighbor_gap *= scale_floor
        if nonneighbor_gap <= 2 * radius:
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

    def _layout(self) -> tuple:
        return (
            self.tile_shape,
            self.axis,
            tuple(type(pattern).__qualname__ for pattern in self.axes),
        )
