"""Logical operator charts with index-based, bounded coordinate generation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .chart import Chart
from .geometry import EuclideanGeometry, Geometry, TorusGeometry
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
        from torchcst._backends.torch.charts.lazy import _indices

        return _indices(self, selection)

    def positions(self, indices: Tensor) -> Tensor:
        from torchcst._backends.torch.charts.lazy import positions

        return positions(self, indices)

    def _embed(self, coordinates: Tensor) -> Tensor:
        from torchcst._backends.torch.charts.lazy import _embed

        return _embed(self, coordinates)

    def squared_distance(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor:
        from torchcst._backends.torch.charts.lazy import squared_distance

        return squared_distance(self, centers, selection)

    def center_offsets(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor:
        from torchcst._backends.torch.charts.lazy import center_offsets

        return center_offsets(self, centers, selection)

    def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
        from torchcst._backends.torch.charts.lazy import initialize_centers

        return initialize_centers(self, atoms, mode=mode)

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


def _torus_line_axis(
    axes: tuple[SitePattern, ...], geometry: Geometry, *, strip_axis: int | None = None
) -> None:
    if not isinstance(geometry, TorusGeometry):
        return
    offset = 0
    for index, pattern in enumerate(axes):
        if offset == geometry.circle_axis and isinstance(pattern, LinePattern):
            if strip_axis is not None and index != strip_axis:
                break
            return
        offset += pattern.dim
    raise ValueError("TorusGeometry.circle_axis must select the Chart LinePattern axis")


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
        if self.geometry.intrinsic_dim != dim:
            raise ValueError(
                "geometry dimensions must match combined axis pattern dimensions"
            )
        _torus_line_axis(axes, self.geometry)
        if isinstance(self.geometry, TorusGeometry):
            offset = 0
            for pattern in axes:
                if offset == self.geometry.circle_axis:
                    low, high = pattern.bounds()
                    if float(high[0] - low[0]) >= self.geometry.circumference:
                        raise ValueError(
                            "torus circle axis must span less than one turn"
                        )
                    break
                offset += pattern.dim
        self.axes = nn.ModuleList(axes)

    @property
    def reference(self) -> Tensor:
        return self.axes[0].reference

    def positions(self, indices: Tensor) -> Tensor:
        from torchcst._backends.torch.charts.product import positions

        return positions(self, indices)

    def _bounds(self) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.charts.product import _bounds

        return _bounds(self)

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
        if self.geometry.intrinsic_dim != dim:
            raise ValueError(
                "geometry dimensions must match combined axis pattern dimensions"
            )
        _torus_line_axis(axes, self.geometry, strip_axis=axis)
        self.axes = nn.ModuleList(axes)
        self.axis = axis
        self.tile_shape = tile_shape
        self.register_buffer("tile_pitch", line.reference.new_tensor(float(tile_pitch)))
        if isinstance(self.geometry, TorusGeometry):
            low, high = self._bounds()
            circular_span = float(
                high[self.geometry.circle_axis] - low[self.geometry.circle_axis]
            )
            if circular_span >= self.geometry.circumference:
                raise ValueError("torus strip axis must span less than one turn")

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
        from torchcst._backends.torch.charts.strip import tile_indices

        return tile_indices(self, station)

    def validate_support(self, radius: float) -> None:
        from torchcst._backends.torch.charts.strip import validate_support

        return validate_support(self, radius)

    def _validate_torus_support(self, radius: float) -> None:
        """Check second-neighbor stations, the nearest nonadjacent pairs."""
        from torchcst._backends.torch.charts.strip import _validate_torus_support

        return _validate_torus_support(self, radius)

    def positions(self, indices: Tensor) -> Tensor:
        from torchcst._backends.torch.charts.strip import positions

        return positions(self, indices)

    def _bounds(self) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.charts.strip import _bounds

        return _bounds(self)

    def _layout(self) -> tuple:
        return (
            self.tile_shape,
            self.axis,
            tuple(type(pattern).__qualname__ for pattern in self.axes),
        )
