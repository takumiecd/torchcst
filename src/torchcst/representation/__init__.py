"""Public surface of coordinate domains and representation specifications."""

from .domains import Box, CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .kernels import (
    CONTINUOUS_KERNELS,
    ContinuousKernel,
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
    "AbsorbPlanStep",
    "AxisSurvey",
    "Box",
    "ChartProposal",
    "ChartSurvey",
    "CONTINUOUS_KERNELS",
    "ContinuousKernel",
    "CoordinateDomain",
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
    "propose_chart",
    "survey_chart",
]
