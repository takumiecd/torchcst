"""Pure layout metadata for the radius-native cell-mass contract."""

from dataclasses import dataclass

from torchcst.charts.regular_grid import RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec

from .parameterizations import PhysicalAmpRadiusSpec, PolarAmpRadiusSpec
from .spec import RadiusIntegratedProfileBinding


@dataclass(frozen=True)
class RadiusCellMassGrid:
    site_count: int
    spacing: float
    origin: float
    binding: RadiusIntegratedProfileBinding

    @property
    def period(self):
        """The chart circumference Q, distinct from each local profile period."""
        return self.site_count * self.spacing


def radius_cell_mass_axes(charts, kernel):
    """Validate declarations without importing Torch or creating buffers."""
    if (
        len(charts) != 2
        or kernel.revision != 7
        or kernel.composition != "separable"
        or type(kernel.parameterization)
        not in (PhysicalAmpRadiusSpec, PolarAmpRadiusSpec)
    ):
        raise ValueError("radius cell masses require revision7 radius coordinates")
    axes = []
    for chart, binding in zip(charts, kernel.profiles):
        if (
            type(chart) is not RegularGridChartSpec
            or len(chart.grid_shape) != 1
            or len(chart.grid_shape[0]) != 1
            or chart.trainable
            or type(chart.geometry) is not FlatTorusGeometrySpec
            or chart.geometry.intrinsic_dim != 1
            or type(binding) is not RadiusIntegratedProfileBinding
        ):
            raise ValueError(
                "radius cell masses require fixed one-dimensional torus grids"
            )
        sites = chart.shape[0]
        if sites % binding.base_tile_sites:
            raise ValueError("base_tile_sites must divide the chart site count")
        ratio = sites // binding.base_tile_sites
        while ratio > 1 and ratio % binding.merge_factor == 0:
            ratio //= binding.merge_factor
        if ratio != 1:
            raise ValueError("chart/base tile ratio must be a power of merge_factor")
        axis = RadiusCellMassGrid(sites, chart.spacing[0], chart.origin[0], binding)
        if chart.geometry.periods != (axis.period,):
            raise ValueError("chart period must equal site_count * spacing")
        axes.append(axis)
    if type(kernel.parameterization) is PolarAmpRadiusSpec:
        for axis, bounds in zip(
            axes,
            (
                kernel.parameterization.input_bounds,
                kernel.parameterization.output_bounds,
            ),
        ):
            if bounds.maximum > axis.period / 2:
                raise ValueError("radius bounds must fit each chart half-period")
    else:
        settings = dict(kernel.initialization.settings)
        floor = dict(kernel.update.settings).get("radius_floor")
        ceiling = min(axis.period for axis in axes) / 2
        if type(floor) not in (float, int) or not 0 < floor <= ceiling:
            raise ValueError("radius_floor must fit the chart half-period")
        if (
            not floor
            <= settings["radius_init_min"]
            <= settings["radius_init_max"]
            <= ceiling
        ):
            raise ValueError("radius initialization must fit the chart half-period")
    return tuple(axes)
