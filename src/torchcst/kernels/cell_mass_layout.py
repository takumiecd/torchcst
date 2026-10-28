"""Pure mathematical layout checks, shared by reference and CUDA metadata."""

from dataclasses import dataclass

from torchcst.charts.regular_grid import RegularGridChartSpec
from torchcst.geometry.spec import FlatTorusGeometrySpec

from .parameterizations import PhysicalAmpWidthSpec, PolarAmpWidthSpec
from .spec import CellIntegratedProfileBinding


@dataclass(frozen=True)
class CellMassGrid:
    site_count: int
    spacing: float
    origin: float
    binding: CellIntegratedProfileBinding

    @property
    def period(self):
        return self.site_count * self.spacing


def cell_mass_axes(charts, kernel):
    """Return input/output grids without importing execution code or tensors."""
    if (
        len(charts) != 2
        or kernel.composition != "separable"
        or type(kernel.parameterization)
        not in (PhysicalAmpWidthSpec, PolarAmpWidthSpec)
    ):
        raise ValueError("cell masses require a physical-sigma or Polar chart pair")
    axes = []
    for chart, binding in zip(charts, kernel.profiles):
        if (
            type(chart) is not RegularGridChartSpec
            or len(chart.grid_shape) != 1
            or len(chart.grid_shape[0]) != 1
            or chart.trainable
            or type(chart.geometry) is not FlatTorusGeometrySpec
            or chart.geometry.intrinsic_dim != 1
            or type(binding) is not CellIntegratedProfileBinding
        ):
            raise ValueError(
                "cell masses require frozen one-dimensional periodic regular grids"
            )
        n = chart.shape[0]
        if n & (n - 1) or binding.min_tile_sites > n:
            raise ValueError("cell-mass grid and minimum tile must be dyadic and fit")
        axis = CellMassGrid(n, chart.spacing[0], chart.origin[0], binding)
        if chart.geometry.periods != (axis.period,):
            raise ValueError("cell-mass chart period must equal site_count * spacing")
        axes.append(axis)
    if type(kernel.parameterization) is PolarAmpWidthSpec:
        for axis, bounds in zip(
            axes,
            (
                kernel.parameterization.input_bounds,
                kernel.parameterization.output_bounds,
            ),
        ):
            if bounds.maximum > axis.period / 3:
                raise ValueError(
                    "Polar bandwidth bounds must fit each cell-mass period/3"
                )
        return tuple(axes)
    floor = dict(kernel.update.settings).get("sigma_floor")
    if (
        type(floor) not in (int, float)
        or not 0 < floor <= min(a.period for a in axes) / 3
    ):
        raise ValueError("sigma_floor must fit the cell-mass width domain")
    return tuple(axes)
