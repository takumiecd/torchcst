"""Public surface of coordinate domains and representation specifications."""

from .domains import Box, CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .factors import (
    CONTINUOUS_FACTORS,
    AtomColumn,
    ContinuousFactor,
    GaborFactor,
    GaussianFactor,
    MaturityGaussianFactor,
    OverlapScale,
    TriangularFactor,
    pairwise_overlap,
)
from .gauges import (
    Amplitude,
    AmplitudeGauge,
    L2NormalizedColumns,
    UnitFootprint,
    require_gauge,
)
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .proposal import ChartProposal, propose_chart
from .spec import RepresentationSpec
from .survey import AxisSurvey, ChartSurvey, survey_chart

__all__ = [
    "CONTINUOUS_FACTORS",
    "AbsorbAssessment",
    "AbsorbPlanStep",
    "Amplitude",
    "AmplitudeGauge",
    "AtomColumn",
    "AxisSurvey",
    "Box",
    "ChartProposal",
    "ChartSurvey",
    "ContinuousFactor",
    "CoordinateDomain",
    "GaborFactor",
    "GaussianFactor",
    "GramService",
    "IntegerGrid",
    "L2NormalizedColumns",
    "MaturityGaussianFactor",
    "OverlapScale",
    "ParameterRole",
    "RepresentationSpec",
    "Role",
    "Sphere",
    "TriangularFactor",
    "UnitFootprint",
    "pairwise_overlap",
    "propose_chart",
    "require_gauge",
    "survey_chart",
]
