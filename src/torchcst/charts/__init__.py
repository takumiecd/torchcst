"""Concrete chart declarations and Tensor owners under a common abstract contract."""

from . import presets
from .base import ChartSpec, ChartState
from .explicit import ExplicitChartSpec, ExplicitChartState
from .periodic_grid import PeriodicGridChartSpec, PeriodicGridChartState
from .product import ProductChartSpec, ProductChartState
from .strip import StripChartSpec, StripChartState

_STATES = {
    ExplicitChartSpec: ExplicitChartState,
    ProductChartSpec: ProductChartState,
    PeriodicGridChartSpec: PeriodicGridChartState,
    StripChartSpec: StripChartState,
}


def compile_chart(spec: ChartSpec, *, device=None, dtype=None) -> ChartState:
    """Construct an exact supported owner at a configuration boundary."""
    state_type = _STATES.get(type(spec))
    if state_type is None:
        raise ValueError("unsupported chart declaration")
    return state_type(spec, device=device, dtype=dtype)


__all__ = [
    "ChartSpec",
    "ChartState",
    "ExplicitChartSpec",
    "ExplicitChartState",
    "PeriodicGridChartSpec",
    "PeriodicGridChartState",
    "ProductChartSpec",
    "ProductChartState",
    "StripChartSpec",
    "StripChartState",
    "compile_chart",
    "presets",
]
