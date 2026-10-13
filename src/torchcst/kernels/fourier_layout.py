"""Pure configuration checks shared by Fourier reference algorithms."""

import math

from torchcst.charts import PeriodicGridChartSpec, RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec
from torchcst.kernels.profiles.fourier import (
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
)


def fourier_tile_layout(chart, kernel):
    if (
        type(chart) not in (PeriodicGridChartSpec, RegularGridChartSpec)
        or type(chart.geometry) is not FlatTorusGeometrySpec
        or chart.revision != 1
        or chart.geometry.revision != 1
        or kernel.composition != "profile_product"
        or kernel.revision != 4
        or kernel.tile_support is None
    ):
        raise ValueError(
            "Fourier tile product requires a FlatTorus periodic/regular grid"
        )
    sizes = (
        chart.coordinate_shape
        if type(chart) is RegularGridChartSpec
        else chart.grid_shape
    )
    widths = kernel.tile_support.tile_widths
    if len(sizes) != len(widths) or len(sizes) != len(kernel.profiles):
        raise ValueError(
            "one Fourier profile and tile width are required per coordinate"
        )
    if kernel.parameterization.input_bounds.maximum > min(widths):
        raise ValueError("maximum width exceeds tile width")
    layout = []
    for dim, (size, width, period, binding) in enumerate(
        zip(sizes, widths, chart.geometry.periods, kernel.profiles)
    ):
        if (
            type(binding.profile)
            not in (ChebyshevSquareSpec, DPSSSquareSpec, KaiserSquareSpec)
            or binding.profile.revision != 1
        ):
            raise ValueError("unsupported periodic Fourier profile")
        tiles = round(period / width)
        if tiles < 3 or not math.isclose(tiles * width, period, rel_tol=1e-6):
            raise ValueError(
                "global period must contain an integral number of at least three tiles"
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
