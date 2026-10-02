"""Metadata-only suitability checks for normalized Euclidean Strip algorithms."""

import math
from dataclasses import dataclass
from functools import cache

from torchcst.geometry.spec import (
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
)
from torchcst.kernels.presets import NORMALIZED_RADIAL_TRIWEIGHT
from torchcst.operators.spec import OperatorSpec, SingleChartSpec

OPERATION = "linear"
SEMANTICS = "normalized-strip-triweight-l2-v1"


@dataclass(frozen=True)
class NormalizedStripGeometry:
    sizes: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    def __post_init__(self):
        if (
            not isinstance(self.sizes, tuple)
            or len(self.sizes) != 3
            or any(type(n) is not int or n <= 0 for n in self.sizes)
        ):
            raise ValueError("sizes must contain three positive integers")
        for name, values in (("origin", self.origin), ("spacing", self.spacing)):
            if (
                not isinstance(values, tuple)
                or len(values) != 3
                or not all(math.isfinite(v) for v in values)
            ):
                raise ValueError(f"{name} must contain three finite values")
        if min(self.spacing) <= 0:
            raise ValueError("spacing must be positive")

    @property
    def n(self):
        return self.sizes[0]

    @property
    def k(self):
        return math.prod(self.sizes[1:])

    @classmethod
    def from_declaration(cls, spec):
        sizes, origin, spacing = normalized_strip_metadata(spec)
        return cls(sizes=sizes, origin=origin, spacing=spacing)


@cache
def geometry(spec):
    """Cache immutable sites only; never cache atom values or autograd state."""
    return NormalizedStripGeometry.from_declaration(spec)


def normalized_strip_metadata(spec):
    """Reject a different operator before it reaches the specialized algorithms."""
    if (
        not isinstance(spec, OperatorSpec)
        or not isinstance(spec.layout, SingleChartSpec)
        or spec.revision != 1
        or spec.kernel != NORMALIZED_RADIAL_TRIWEIGHT
    ):
        raise ValueError("operator differs from the normalized Strip kernel contract")
    chart = spec.layout.chart
    if (
        chart.kind not in ("product", "strip")
        or chart.geometry != EuclideanGeometrySpec(intrinsic_dim=3)
        or chart.trainable
        or chart.revision != 1
        or len(chart.axes) != 2
        or type(chart.axes[0]) is not LinePatternSpec
        or type(chart.axes[1]) is not GridPatternSpec
        or any(axis.revision != 1 for axis in chart.axes)
        or chart.axes[1].dim != 2
    ):
        raise ValueError("normalized Strip requires fixed regular 3D Euclidean sites")
    line, grid = chart.axes
    if chart.kind == "strip" and (
        chart.axis != 0 or chart.tile_pitch != chart.tile_shape[0] * line.spacing[0]
    ):
        raise ValueError("normalized Strip requires contiguous row sites")
    sizes = (chart.shape[0], *grid.shape)
    origin = (*line.start, *grid.start)
    spacing = (*line.spacing, *grid.spacing)
    return sizes, origin, spacing
