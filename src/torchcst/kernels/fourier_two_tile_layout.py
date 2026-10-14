"""Pure configuration checks for the revision-5 two-tile family."""

import math

from torchcst.charts import PeriodicGridChartSpec, RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels.profiles.fourier import KaiserTemplateSpec
from torchcst.kernels.spec import FourierTwoTileKernelSpec
from torchcst.kernels.support import FourierTwoTileSupportSpec


def two_tile_layout(chart, kernel):
    """Return (tile count, sites per tile, physical width) for each axis."""
    if (
        type(chart) not in (PeriodicGridChartSpec, RegularGridChartSpec)
        or type(chart.geometry) is not FlatTorusGeometrySpec
        or chart.revision != 1
        or chart.geometry.revision != 1
        or type(kernel) is not FourierTwoTileKernelSpec
        or kernel.composition != "profile_product"
        or kernel.revision != 5
        or type(kernel.tile_support) is not FourierTwoTileSupportSpec
    ):
        raise ValueError("two-tile templates require a FlatTorus periodic/regular grid")
    sizes = (
        chart.coordinate_shape
        if type(chart) is RegularGridChartSpec
        else chart.grid_shape
    )
    widths = kernel.tile_support.tile_widths
    if len(sizes) != len(widths) or len(sizes) != len(kernel.profiles):
        raise ValueError("one two-tile profile and width are required per coordinate")
    bounds = kernel.parameterization.input_bounds
    layout = []
    for dim, (size, width, period, binding) in enumerate(
        zip(sizes, widths, chart.geometry.periods, kernel.profiles)
    ):
        spec = binding.profile
        if type(spec) is not KaiserTemplateSpec or spec.revision != 1:
            raise ValueError("unsupported two-tile template profile")
        if bounds.maximum > width / 2 and not math.isclose(
            bounds.maximum, width / 2, rel_tol=1e-6
        ):
            raise ValueError("maximum width exceeds half the tile width")
        if (
            (
                spec.width_knots[0] > bounds.minimum
                and not math.isclose(spec.width_knots[0], bounds.minimum, rel_tol=1e-6)
            )
            or (
                spec.width_knots[-1] < bounds.maximum
                and not math.isclose(spec.width_knots[-1], bounds.maximum, rel_tol=1e-6)
            )
            or spec.width_knots[-1] > width / 2
        ):
            raise ValueError(
                "width knots must cover bandwidth bounds and stay within half a tile"
            )
        # Kernel scalar buffers may round declared bounds when Module.to()
        # casts them to FP32. The profile clamps interpolation to its immutable
        # FP64 knots; this metadata allowance does not relax numerical gates.
        tiles = round(period / width)
        if tiles < 2 or not math.isclose(tiles * width, period, rel_tol=1e-6):
            raise ValueError(
                "global period must contain an integral number of at least two tiles"
            )
        if size % tiles:
            raise ValueError("each tile must contain a whole number of uniform sites")
        if type(chart) is RegularGridChartSpec and not math.isclose(
            chart.spacing[dim] * size, period, rel_tol=1e-6
        ):
            raise ValueError(
                "regular grid spacing must cover one complete global period"
            )
        layout.append((tiles, size // tiles, width))
    return tuple(layout)
