"""Logical operator charts with index-based, bounded coordinate generation."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from .chart import Chart
from .geometry import EuclideanGeometry, Geometry
from .pattern import LinePattern, SitePattern


class _LazyChart(Chart):
    """A Chart whose sites are computed only for requested flat indices."""

    def __init__(self, shape: tuple[int, int], geometry: Geometry) -> None:
        nn.Module.__init__(self)
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a Geometry")
        self._shape = shape
        self.geometry = geometry

    @property
    def shape(self) -> tuple[int, int]:
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


class ProductChart(_LazyChart):
    """Lazy Cartesian layout of output and input site patterns."""

    def __init__(
        self,
        output: SitePattern,
        input: SitePattern,
        *,
        geometry: Geometry | None = None,
    ) -> None:
        if not isinstance(output, SitePattern) or not isinstance(input, SitePattern):
            raise TypeError("output and input must be SitePattern instances")
        if (
            output.reference.device != input.reference.device
            or output.reference.dtype != input.reference.dtype
        ):
            raise ValueError("output and input patterns must share device and dtype")
        dim = output.dim + input.dim
        super().__init__(
            (output.features, input.features), geometry or EuclideanGeometry(dim)
        )
        if self.embedding_dim != dim:
            raise ValueError(
                "geometry.embedding_dim must match combined pattern dimensions"
            )
        self.output = output
        self.input = input

    @property
    def reference(self) -> Tensor:
        return self.output.reference

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
        output_index = torch.div(indices, self.shape[1], rounding_mode="floor")
        input_index = indices % self.shape[1]
        return torch.cat(
            (self.output.positions(output_index), self.input.positions(input_index)),
            dim=-1,
        )

    def _bounds(self) -> tuple[Tensor, Tensor]:
        out_low, out_high = self.output.bounds()
        in_low, in_high = self.input.bounds()
        return torch.cat((out_low, in_low)), torch.cat((out_high, in_high))

    def _layout(self) -> tuple:
        return (type(self.output).__qualname__, type(self.input).__qualname__)


class StripChart(_LazyChart):
    """Lay matrix tiles on a line, with independent coordinates inside each tile.

    ``tile_pitch`` is the distance between neighboring tile stations. A kernel
    with support radius strictly below this pitch can touch at most two tile
    stations. ``seam_gap`` adds distance when the sweep starts a new row.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        tile_shape: tuple[int, int],
        *,
        tile_pitch: float,
        local_output: SitePattern | None = None,
        local_input: SitePattern | None = None,
        sweep: Literal["input", "output"] = "input",
        snake: bool = True,
        seam_gap: float = 0.0,
        seam_policy: Literal["join", "separate"] = "join",
        geometry: Geometry | None = None,
    ) -> None:
        if (
            len(shape) != 2
            or len(tile_shape) != 2
            or any(type(n) is not int or n < 1 for n in (*shape, *tile_shape))
        ):
            raise ValueError(
                "shape and tile_shape must each contain two positive integers"
            )
        shape = tuple(shape)
        tile_shape = tuple(tile_shape)
        if sweep not in ("input", "output") or not isinstance(snake, bool):
            raise ValueError("sweep must be 'input' or 'output' and snake must be bool")
        if seam_policy not in ("join", "separate"):
            raise ValueError("seam_policy must be 'join' or 'separate'")
        if (
            not math.isfinite(tile_pitch)
            or tile_pitch <= 0
            or not math.isfinite(seam_gap)
            or seam_gap < 0
        ):
            raise ValueError("tile_pitch must be positive and seam_gap nonnegative")
        local_output = local_output or LinePattern(tile_shape[0], spacing=1.0)
        local_input = local_input or LinePattern(tile_shape[1], spacing=1.0)
        if (
            local_output.features != tile_shape[0]
            or local_input.features != tile_shape[1]
        ):
            raise ValueError("local pattern sizes must match tile_shape")
        if (
            local_output.reference.device != local_input.reference.device
            or local_output.reference.dtype != local_input.reference.dtype
        ):
            raise ValueError("local patterns must share device and dtype")
        dim = 1 + local_output.dim + local_input.dim
        super().__init__(shape, geometry or EuclideanGeometry(dim))
        if self.embedding_dim != dim:
            raise ValueError(
                "geometry.embedding_dim must match strip coordinate dimensions"
            )
        self.tile_shape = tile_shape
        self.sweep = sweep
        self.snake = snake
        self.seam_policy = seam_policy
        self.local_output = local_output
        self.local_input = local_input
        self.register_buffer(
            "tile_pitch", local_output.reference.new_tensor(float(tile_pitch))
        )
        self.register_buffer(
            "seam_gap", local_output.reference.new_tensor(float(seam_gap))
        )

    @property
    def reference(self) -> Tensor:
        return self.tile_pitch

    @property
    def tile_grid(self) -> tuple[int, int]:
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_shape))

    @property
    def tile_count(self) -> int:
        return math.prod(self.tile_grid)

    def tile_indices(self, station: int) -> tuple[Tensor, Tensor]:
        """Logical and tile-local indices for one physical tile station.

        Edge tiles omit positions outside the logical weight. Each returned
        array is at most one tile long; no global indirection table is stored.
        """

        if type(station) is not int or not 0 <= station < self.tile_count:
            raise IndexError("tile station out of bounds")
        rows, cols = self.tile_grid
        inner_count = cols if self.sweep == "input" else rows
        outer, inner = divmod(station, inner_count)
        if self.snake and outer % 2:
            inner = inner_count - 1 - inner
        tile_row, tile_col = (outer, inner) if self.sweep == "input" else (inner, outer)
        local_row = torch.arange(self.tile_shape[0], device=self.device)
        local_col = torch.arange(self.tile_shape[1], device=self.device)
        row = tile_row * self.tile_shape[0] + local_row[:, None]
        col = tile_col * self.tile_shape[1] + local_col[None, :]
        valid = (row < self.shape[0]) & (col < self.shape[1])
        logical = (row * self.shape[1] + col).expand(self.tile_shape)
        local = torch.arange(math.prod(self.tile_shape), device=self.device).reshape(
            self.tile_shape
        )
        return logical[valid], local[valid]

    def validate_support(self, radius: float) -> None:
        if not isinstance(self.geometry, EuclideanGeometry):
            raise NotImplementedError("strip support bounds require EuclideanGeometry")
        if not math.isfinite(radius) or radius <= 0 or radius >= float(self.tile_pitch):
            raise ValueError(
                "kernel support radius must be positive and smaller than tile_pitch"
            )
        outer_count = self.tile_grid[0] if self.sweep == "input" else self.tile_grid[1]
        if (
            self.seam_policy == "separate"
            and outer_count > 1
            and float(self.tile_pitch + self.seam_gap) <= 2 * radius
        ):
            raise ValueError(
                "separate seams require tile_pitch + seam_gap > 2 * support radius"
            )

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
        row = torch.div(indices, self.shape[1], rounding_mode="floor")
        col = indices % self.shape[1]
        tile_row = torch.div(row, self.tile_shape[0], rounding_mode="floor")
        tile_col = torch.div(col, self.tile_shape[1], rounding_mode="floor")
        rows, cols = self.tile_grid
        if self.sweep == "input":
            outer, inner, inner_count = tile_row, tile_col, cols
        else:
            outer, inner, inner_count = tile_col, tile_row, rows
        if self.snake:
            inner = torch.where(outer % 2 == 0, inner, inner_count - 1 - inner)
        station = outer * inner_count + inner
        longitude = (
            station.to(self.dtype) * self.tile_pitch
            + outer.to(self.dtype) * self.seam_gap
        )
        return torch.cat(
            (
                longitude[:, None],
                self.local_output.positions(row % self.tile_shape[0]),
                self.local_input.positions(col % self.tile_shape[1]),
            ),
            dim=-1,
        )

    def _bounds(self) -> tuple[Tensor, Tensor]:
        out_low, out_high = self.local_output.bounds()
        in_low, in_high = self.local_input.bounds()
        major_count = self.tile_grid[0] if self.sweep == "input" else self.tile_grid[1]
        stations = math.prod(self.tile_grid)
        start = self.tile_pitch.new_zeros(1)
        end = (stations - 1) * self.tile_pitch.reshape(1) + (
            major_count - 1
        ) * self.seam_gap.reshape(1)
        return torch.cat((start, out_low, in_low)), torch.cat((end, out_high, in_high))

    def _layout(self) -> tuple:
        return (
            self.tile_shape,
            self.sweep,
            self.snake,
            self.seam_policy,
            type(self.local_output).__qualname__,
            type(self.local_input).__qualname__,
        )
