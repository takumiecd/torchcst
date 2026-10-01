"""Backend-independent linear operator declarations."""

from dataclasses import dataclass, field

from torchcst.geometry.spec import ChartSpec
from torchcst.kernels.spec import KernelSpec


@dataclass(frozen=True, kw_only=True)
class SingleChartSpec:
    """One chart whose logical axes are [output, input]."""

    chart: ChartSpec

    def __post_init__(self):
        if not isinstance(self.chart, ChartSpec):
            raise TypeError("chart must be a ChartSpec")
        if len(self.chart.shape) != 2:
            raise ValueError("a single chart must describe an [out, in] operator")

    @property
    def charts(self):
        return (self.chart,)

    @property
    def shape(self):
        return self.chart.shape


@dataclass(frozen=True, kw_only=True)
class ChartPairSpec:
    """Independent input/output observation charts; retained for current kernels."""

    input_chart: ChartSpec
    output_chart: ChartSpec

    def __post_init__(self):
        if not all(isinstance(c, ChartSpec) for c in self.charts):
            raise TypeError("input_chart and output_chart must be ChartSpec instances")

    @property
    def charts(self):
        return (self.input_chart, self.output_chart)

    @property
    def shape(self):
        return (self.output_chart.features, self.input_chart.features)


@dataclass(frozen=True, kw_only=True)
class OperatorSpec:
    """A sum of kernel atoms with an explicit observation layout.

    Tensor state, backend choices and launch settings are supplied separately.
    This stage describes linear maps; other operations need their own contract.
    """

    layout: SingleChartSpec | ChartPairSpec
    kernel: KernelSpec
    operation_id: str = field(default="linear", init=False)
    revision: int = 1

    def __post_init__(self):
        if not isinstance(self.layout, (SingleChartSpec, ChartPairSpec)):
            raise TypeError("choose a single-chart or chart-pair layout")
        if not isinstance(self.kernel, KernelSpec):
            raise TypeError("kernel must be a KernelSpec")
        if type(self.revision) is not int or self.revision != 1:
            raise ValueError("unsupported linear operator revision")
        kernel = self.kernel
        while kernel.composition == "amplitude":
            kernel = kernel.inner
        required = "radial" if len(self.charts) == 1 else "separable"
        if kernel.composition != required:
            raise ValueError("kernel composition differs from the chart layout")

    @property
    def charts(self):
        return self.layout.charts

    @property
    def shape(self):
        return self.layout.shape

    @property
    def out_features(self):
        return self.shape[0]

    @property
    def in_features(self):
        return self.shape[1]

    def bind(self, *, charts, kernel, atoms):
        """Validate existing live state at configuration time, without copying it."""
        from .binding import Operator

        operator = Operator(charts=charts, kernel=kernel, atoms=atoms)
        if operator.declaration() != self:
            raise ValueError("live operator settings differ from the declaration")
        return operator
