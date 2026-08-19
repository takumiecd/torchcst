"""Public surface of coordinate domains and representation specifications."""

from .domains import Box, CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .gauges import Amplitude, AmplitudeGauge, UnitFootprint, require_gauge
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .kernels import (
    CONTINUOUS_KERNELS,
    AtomColumn,
    ContinuousKernel,
    GaborKernel,
    GaussianKernel,
    OverlapScale,
    pairwise_overlap,
    TriangularKernel,
)
from .spec import RepresentationSpec
from .proposal import ChartProposal, propose_chart
from .survey import AxisSurvey, ChartSurvey, survey_chart

__all__ = [
    "AbsorbAssessment",
    "Amplitude",
    "AmplitudeGauge",
    "AbsorbPlanStep",
    "AxisSurvey",
    "Box",
    "ChartProposal",
    "ChartSurvey",
    "CONTINUOUS_KERNELS",
    "AtomColumn",
    "ContinuousKernel",
    "CoordinateDomain",
    "GaborKernel",
    "GaussianKernel",
    "OverlapScale",
    "pairwise_overlap",
    "GramService",
    "IntegerGrid",
    "ParameterRole",
    "RepresentationSpec",
    "Role",
    "Sphere",
    "TriangularKernel",
    "UnitFootprint",
    "require_gauge",
    "propose_chart",
    "survey_chart",
]
