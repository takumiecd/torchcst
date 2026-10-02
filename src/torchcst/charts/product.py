"""Cartesian-product declaration and bounded axis Tensor ownership."""

from dataclasses import dataclass, field, replace

from torchcst.geometry.spec import TorusGeometrySpec
from torchcst.patterns.spec import PatternSpec

from .base import (
    ChartSpec,
    _AxisChartState,
    _circle_axis,
    _validate_axes,
    _validate_torus_span,
)


@dataclass(frozen=True, kw_only=True)
class ProductChartSpec(ChartSpec):
    axes: tuple[PatternSpec, ...]
    kind: str = field(default="product", init=False)

    def __post_init__(self):
        super().__post_init__()
        _validate_axes(self)
        if isinstance(self.geometry, TorusGeometrySpec):
            axis = _circle_axis(self)
            _validate_torus_span(
                self, (self.shape[axis] - 1) * self.axes[axis].spacing[0]
            )


class ProductChartState(_AxisChartState):
    spec_type = ProductChartSpec

    def declaration(self) -> ProductChartSpec:
        return replace(self.spec, **self._declaration_values())
