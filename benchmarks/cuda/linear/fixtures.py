"""Pure regular-site fixtures shared by registry tests and benchmarks."""

import math

from torchcst.geometry.spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
)
from torchcst.kernels.presets import NORMALIZED_RADIAL_TRIWEIGHT
from torchcst.operators.spec import OperatorSpec, SingleChartSpec


def operator_spec(*, sizes, origin, spacing):
    """Canonical contiguous sites; CUDA storage tiling is not mathematical state."""
    chart = ChartSpec(
        kind="product",
        geometry=EuclideanGeometrySpec(intrinsic_dim=3),
        shape=(sizes[0], math.prod(sizes[1:])),
        axes=(
            LinePatternSpec(
                shape=(sizes[0],), start=(origin[0],), spacing=(spacing[0],)
            ),
            GridPatternSpec(shape=sizes[1:], start=origin[1:], spacing=spacing[1:]),
        ),
    )
    return OperatorSpec(
        layout=SingleChartSpec(chart=chart), kernel=NORMALIZED_RADIAL_TRIWEIGHT
    )
