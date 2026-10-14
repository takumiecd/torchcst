"""Metadata-only layout checks for the revision6 periodic mixture family."""

import math

from torchcst.charts import PeriodicGridChartSpec, RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec

from .profiles.matern import MaternMixtureSpec


def matern_layout(chart, kernel):
    if (
        type(chart) not in (PeriodicGridChartSpec, RegularGridChartSpec)
        or type(chart.geometry) is not FlatTorusGeometrySpec
        or chart.revision != 1
        or chart.geometry.revision != 1
        or kernel.revision != 6
        or kernel.composition != "profile_product"
    ):
        raise ValueError("mixtures require revision6 and a flat periodic/regular grid")
    sizes = (
        chart.coordinate_shape
        if type(chart) is RegularGridChartSpec
        else chart.grid_shape
    )
    if len(sizes) != len(kernel.profiles):
        raise ValueError("one mixture profile is required per coordinate")
    bounds = kernel.parameterization.input_bounds
    layout = []
    for dim, (size, period, binding) in enumerate(
        zip(sizes, chart.geometry.periods, kernel.profiles)
    ):
        spec = binding.profile
        if type(spec) is not MaternMixtureSpec or spec.revision != 1:
            raise ValueError("unsupported mixture profile revision")
        for actual, endpoint, lower in [
            (bounds.minimum, spec.width_knots[0], True),
            (bounds.maximum, spec.width_knots[-1], False),
        ]:
            if (actual < endpoint if lower else actual > endpoint) and not math.isclose(
                actual, endpoint, rel_tol=1e-6
            ):
                raise ValueError("width knots must cover bandwidth bounds")
        spacing = (
            chart.spacing[dim] if type(chart) is RegularGridChartSpec else period / size
        )
        if not math.isclose(size * spacing, period, rel_tol=1e-6):
            raise ValueError("uniform sites must cover one complete global period")
        layout.append((size, period, spacing))
    return tuple(layout)
