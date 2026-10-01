"""The existing normalized Strip contract expressed as a single-chart operator."""

import math

from torchcst.geometry.spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
)
from torchcst.kernels.normalization import NormalizationSpec
from torchcst.kernels.parameterizations.log_width import LogWidthSpec
from torchcst.kernels.profiles import TriweightSpec
from torchcst.kernels.spec import KernelSpec, ProfileBinding, StatePolicySpec

from .spec import OperatorSpec, SingleChartSpec

NORMALIZED_STRIP_KERNEL = KernelSpec(
    composition="radial",
    profiles=(
        ProfileBinding(
            profile=TriweightSpec(),
            normalization=NormalizationSpec(
                kind="discrete_l2", domain="operator_sites", floor=1e-6
            ),
        ),
    ),
    parameterization=LogWidthSpec(sigma_min=0.03, sigma_max=3.25),
    initialization=StatePolicySpec(id="provided_atoms"),
    update=StatePolicySpec(id="euclidean"),
)


def normalized_strip_declaration(*, sizes, origin, spacing):
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
        layout=SingleChartSpec(chart=chart), kernel=NORMALIZED_STRIP_KERNEL
    )


def normalized_strip_metadata(spec):
    """Reject a different operator before it reaches the specialized algorithms."""
    if (
        not isinstance(spec, OperatorSpec)
        or not isinstance(spec.layout, SingleChartSpec)
        or spec.kernel != NORMALIZED_STRIP_KERNEL
    ):
        raise ValueError("operator differs from the normalized Strip kernel contract")
    chart = spec.layout.chart
    if (
        chart.kind not in ("product", "strip")
        or chart.geometry != EuclideanGeometrySpec(intrinsic_dim=3)
        or chart.trainable
        or chart.revision != 1
        or type(chart.axes[0]) is not LinePatternSpec
        or type(chart.axes[1]) is not GridPatternSpec
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
