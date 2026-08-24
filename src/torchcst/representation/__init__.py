"""Public surface of coordinate domains and representation specifications."""

from .domains import Box, CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .gauges import (
    Amplitude,
    AmplitudeGauge,
    L2NormalizedColumns,
    UnitFootprint,
    require_gauge,
)
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .kernels import (
    CONTINUOUS_KERNELS,
    AtomColumn,
    ContinuousKernel,
    GaborKernel,
    GaussianKernel,
    MaturityGaussianKernel,
    OverlapScale,
    TriangularKernel,
    pairwise_overlap,
)
from .proposal import ChartProposal, propose_chart
from .spec import RepresentationSpec
from .survey import AxisSurvey, ChartSurvey, survey_chart

__all__ = [
    "CONTINUOUS_KERNELS",
    "AbsorbAssessment",
    "AbsorbPlanStep",
    "Amplitude",
    "AmplitudeGauge",
    "AtomColumn",
    "AxisSurvey",
    "Box",
    "ChartProposal",
    "ChartSurvey",
    "ContinuousKernel",
    "CoordinateDomain",
    "GaborKernel",
    "GaussianKernel",
    "GramService",
    "IntegerGrid",
    "L2NormalizedColumns",
    "MaturityGaussianKernel",
    "OverlapScale",
    "ParameterRole",
    "RepresentationSpec",
    "Role",
    "Sphere",
    "TriangularKernel",
    "UnitFootprint",
    "pairwise_overlap",
    "propose_chart",
    "require_gauge",
    "survey_chart",
]
