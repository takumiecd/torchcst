"""Public surface of coordinate domains and representation specifications."""

from .domains import Box, CoordinateDomain, IntegerGrid, ParameterRole, Role, Sphere
from .gram import AbsorbAssessment, AbsorbPlanStep, GramService
from .kernels import (
    CONTINUOUS_KERNELS,
    ContinuousKernel,
    GaussianKernel,
    TriangularKernel,
)
from .spec import RepresentationSpec

__all__ = [
    "AbsorbAssessment",
    "AbsorbPlanStep",
    "Box",
    "CONTINUOUS_KERNELS",
    "ContinuousKernel",
    "CoordinateDomain",
    "GaussianKernel",
    "GramService",
    "IntegerGrid",
    "ParameterRole",
    "RepresentationSpec",
    "Role",
    "Sphere",
    "TriangularKernel",
]
