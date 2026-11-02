"""Cartesian-product declaration and bounded axis Tensor ownership."""

from dataclasses import dataclass, field, replace

from torchcst.patterns.spec import PatternSpec

from .base import (
    ChartSpec,
    _AxisChartState,
    _validate_axes,
)


@dataclass(frozen=True, kw_only=True)
class ProductChartSpec(ChartSpec):
    axes: tuple[PatternSpec, ...]
    kind: str = field(default="product", init=False)

    def __post_init__(self):
        super().__post_init__()
        _validate_axes(self)


class ProductChartState(_AxisChartState):
    spec_type = ProductChartSpec

    def declaration(self) -> ProductChartSpec:
        return replace(self.spec, **self._declaration_values())
