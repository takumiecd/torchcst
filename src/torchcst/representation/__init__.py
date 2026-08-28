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
    PullbackStructure,
    UnitFootprint,
    require_gauge,
)
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .proposal import ChartProposal, propose_chart
from .spec import RepresentationSpec
from .spectrum import (
    FactorizedSpectrumSurvey,
    SpectrumSurvey,
    survey_factorized_map,
    survey_spectrum,
)
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
    "FactorizedSpectrumSurvey",
    "GaborFactor",
    "GaussianFactor",
    "GramService",
    "IntegerGrid",
    "L2NormalizedColumns",
    "MaturityGaussianFactor",
    "OverlapScale",
    "ParameterRole",
    "PullbackStructure",
    "RepresentationSpec",
    "Role",
    "SpectrumSurvey",
    "Sphere",
    "TriangularFactor",
    "UnitFootprint",
    "pairwise_overlap",
    "propose_chart",
    "require_gauge",
    "survey_chart",
    "survey_factorized_map",
    "survey_spectrum",
]
