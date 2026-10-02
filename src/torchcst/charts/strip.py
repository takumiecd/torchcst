"""Strip declaration and ownership of its physical pitch and axis patterns."""

import math
from dataclasses import dataclass, field, replace

import torch

from torchcst._coordinate_state import _floating_dtype
from torchcst._validation import _positive
from torchcst.geometry.spec import TorusGeometrySpec
from torchcst.patterns.spec import LinePatternSpec, PatternSpec

from .base import (
    ChartSpec,
    _AxisChartState,
    _circle_axis,
    _validate_axes,
    _validate_torus_span,
)


@dataclass(frozen=True, kw_only=True)
class StripChartSpec(ChartSpec):
    axes: tuple[PatternSpec, ...]
    tile_shape: tuple[int, ...]
    axis: int
    tile_pitch: float
    kind: str = field(default="strip", init=False)

    def __post_init__(self):
        super().__post_init__()
        _validate_axes(self)
        if (
            not isinstance(self.tile_shape, tuple)
            or len(self.tile_shape) != len(self.shape)
            or any(
                type(t) is not int or not 1 <= t <= n
                for t, n in zip(self.tile_shape, self.shape)
            )
        ):
            raise ValueError("strip tile_shape must fit chart shape")
        if (
            type(self.axis) is not int
            or not 0 <= self.axis < len(self.shape)
            or not isinstance(self.axes[self.axis], LinePatternSpec)
        ):
            raise ValueError("strip axis must select a line pattern")
        if any(
            t != n
            for i, (t, n) in enumerate(zip(self.tile_shape, self.shape))
            if i != self.axis
        ):
            raise ValueError("only the strip axis may be tiled")
        _positive(self.tile_pitch, "tile_pitch")
        span = self.axes[self.axis].spacing[0] * (self.tile_shape[self.axis] - 1)
        if (
            self.shape[self.axis] > self.tile_shape[self.axis]
            and self.tile_pitch <= span
        ):
            raise ValueError("strip tiles must have disjoint physical intervals")
        if isinstance(self.geometry, TorusGeometrySpec):
            axis = _circle_axis(self)
            if axis != self.axis:
                raise ValueError("torus circle_axis must select the chart line axis")
            count, tile = self.shape[axis], self.tile_shape[axis]
            span = ((count - 1) // tile) * self.tile_pitch + (
                (count - 1) % tile
            ) * self.axes[axis].spacing[0]
            _validate_torus_span(self, span)


class StripChartState(_AxisChartState):
    spec_type = StripChartSpec

    def __init__(self, spec, *, device=None, dtype=None):
        super().__init__(spec, device=device, dtype=dtype)
        self.axis, self.tile_shape = spec.axis, spec.tile_shape
        self.register_buffer(
            "tile_pitch",
            torch.tensor(spec.tile_pitch, device=device, dtype=_floating_dtype(dtype)),
        )

    @property
    def tile_grid(self):
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_shape))

    @property
    def tile_count(self):
        return math.prod(self.tile_grid)

    def declaration(self) -> StripChartSpec:
        return replace(
            self.spec,
            **self._declaration_values(),
            tile_pitch=float(self.tile_pitch.detach()),
        )

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "tile_shape": self.tile_shape,
            "axis": self.axis,
        }
