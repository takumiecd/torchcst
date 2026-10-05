"""Fixed-shape continuous operators for PyTorch."""

from ._backends.algorithm import Algorithm
from ._backends.catalog import make_registry as make_algorithm_registry
from ._backends.dispatch import (
    Dispatcher,
    ExecutionBinding,
    FixedSelector,
    OrderedSelector,
    Selector,
)
from ._backends.registry import Registry
from ._backends.schema import DefaultRecipe, ExecutionPlan, SupportResult
from ._backends.state import AlgorithmState
from .atoms import AtomOptimizerState, Atoms, AtomState, OptimizerFieldSpec
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
from .nn import CSTConv2d, CSTLinear, CSTModule
from .operators import ChartPairSpec, Operator, OperatorSpec, SingleChartSpec
from .operators.execution import LinearBinding, LinearInputs
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
    "Algorithm",
    "AlgorithmState",
    "AmpWidthSpec",
    "AtomOptimizerState",
    "AtomState",
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
    "DefaultRecipe",
    "DirectAmpWidthSpec",
    "Dispatcher",
    "EuclideanGeometrySpec",
    "ExecutionBinding",
    "ExecutionPlan",
    "ExplicitChartSpec",
    "ExplicitChartState",
    "FixedSelector",
    "FixedWidthSpec",
    "GaussianSpec",
    "GeometrySpec",
    "GeometryState",
    "GridPatternSpec",
    "KernelOptions",
    "KernelSpec",
    "LinePatternSpec",
    "LinearBinding",
    "LinearInputs",
    "LogWidthSpec",
    "NormalizationSpec",
    "Operator",
    "OperatorSpec",
    "OptimizerFieldSpec",
    "OptimizerStateAdapter",
    "OrderedSelector",
    "PatternSpec",
    "PatternState",
    "PointsPatternSpec",
    "PolarAmpWidthSpec",
    "ProductChartSpec",
    "ProductChartState",
    "ProfileBinding",
    "Registry",
    "Selector",
    "SingleChartSpec",
    "SphereGeometrySpec",
    "StripChartSpec",
    "StripChartState",
    "SupportResult",
    "TorusGeometrySpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
    "chart_presets",
    "compile_chart",
    "geometry_presets",
    "make_algorithm_registry",
    "pattern_presets",
    "presets",
]
