"""Pure regular-site fixtures shared by registry tests and benchmarks."""

import math

from torchcst.charts import ProductChartSpec
from torchcst.geometry.spec import EuclideanGeometrySpec
from torchcst.kernels.factories import NORMALIZED_RADIAL_TRIWEIGHT
from torchcst.operators.spec import OperatorSpec, SingleChartSpec
from torchcst.patterns import GridPatternSpec, LinePatternSpec


def operator_spec(*, sizes, origin, spacing):
    """Canonical contiguous sites; CUDA storage tiling is not mathematical state."""
    chart = ProductChartSpec(
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


def normalized_chart(
    sizes=(64, 4, 4), origin=(0.0, 0.0, 0.0), *, dtype=None, device="cpu"
):
    """One continuous Product chart, using public declarations and no test helpers."""
    import torch

    from torchcst import chart_presets as layouts
    from torchcst import compile_chart
    from torchcst import pattern_presets as patterns

    n, h, j = sizes
    spec = layouts.product(
        shape=(n, h * j),
        axes=(
            patterns.line(n, low=origin[0], high=origin[0] + n - 1),
            patterns.grid(
                (h, j),
                low=origin[1:],
                high=(origin[1] + (h - 1) * 0.5, origin[2] + (j - 1) * 0.5),
            ),
        ),
    )
    return compile_chart(spec, device=device, dtype=dtype or torch.float64)
