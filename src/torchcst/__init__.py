"""Fixed-shape continuous operators for PyTorch."""

from .atoms import Atoms
from .charts import (
    ChartSpec,
    ChartState,
    ExplicitChartSpec,
    ExplicitChartState,
    ProductChartSpec,
    ProductChartState,
    StripChartSpec,
    StripChartState,
    compile_chart,
)
from .charts import presets as chart_presets
from .geometry import (
    EuclideanGeometrySpec,
    GeometrySpec,
    GeometryState,
    SphereGeometrySpec,
    TorusGeometrySpec,
)
from .geometry import presets as geometry_presets
from .kernels import (
    AmpWidthSpec,
    BandwidthBounds,
    BiweightSpec,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    GaussianSpec,
    KernelOptions,
    KernelSpec,
    LogWidthSpec,
    NormalizationSpec,
    PolarAmpWidthSpec,
    ProfileBinding,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
    presets,
)
from .nn import CSTConv2d, CSTLinear, CSTModule, LinearOptions
from .operators import ChartPairSpec, Operator, OperatorSpec, SingleChartSpec
from .optim import CSTOptimizer, OptimizerStateAdapter
from .patterns import (
    GridPatternSpec,
    LinePatternSpec,
    PatternSpec,
    PatternState,
    PointsPatternSpec,
)
from .patterns import presets as pattern_presets

__all__ = [
    "AmpWidthSpec",
    "Atoms",
    "BandwidthBounds",
    "BiweightSpec",
    "CSTConv2d",
    "CSTLinear",
    "CSTModule",
    "CSTOptimizer",
    "ChartPairSpec",
    "ChartSpec",
    "ChartState",
    "DirectAmpWidthSpec",
    "EuclideanGeometrySpec",
    "ExplicitChartSpec",
    "ExplicitChartState",
    "FixedWidthSpec",
    "GaussianSpec",
    "GeometrySpec",
    "GeometryState",
    "GridPatternSpec",
    "KernelOptions",
    "KernelSpec",
    "LinePatternSpec",
    "LinearOptions",
    "LogWidthSpec",
    "NormalizationSpec",
    "Operator",
    "OperatorSpec",
    "OptimizerStateAdapter",
    "PatternSpec",
    "PatternState",
    "PointsPatternSpec",
    "PolarAmpWidthSpec",
    "ProductChartSpec",
    "ProductChartState",
    "ProfileBinding",
    "SingleChartSpec",
    "SphereGeometrySpec",
    "StripChartSpec",
    "StripChartState",
    "TorusGeometrySpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
    "chart_presets",
    "compile_chart",
    "geometry_presets",
    "pattern_presets",
    "presets",
]
