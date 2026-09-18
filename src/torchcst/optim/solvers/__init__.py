"""Numerical solvers for CST local objectives."""

from .base import NormalizedSolver, QuarticSolver
from .device import DeviceBFGS
from .nd import (
    NormalizedBoxFixedPointSolver,
    NormalizedEvaluation,
    NormalizedFixedPointSolver,
    NormalizedSolveResult,
    NormalizedUpdateProblem,
    QuadraticBoxGradientSolver,
    QuadraticGradientSolver,
)
from .newton import BallNewton
from .projected import ProjectedLBFGS
from .quartic import FullQuartic, QuarticSolveResult
from .ray import DeviceRay
from .subspace import SubspaceQuartic
from .taylor import QuadraticTrustSolver

__all__ = [
    "BallNewton",
    "DeviceBFGS",
    "DeviceRay",
    "FullQuartic",
    "NormalizedBoxFixedPointSolver",
    "NormalizedEvaluation",
    "NormalizedFixedPointSolver",
    "NormalizedSolveResult",
    "NormalizedSolver",
    "NormalizedUpdateProblem",
    "ProjectedLBFGS",
    "QuadraticBoxGradientSolver",
    "QuadraticGradientSolver",
    "QuadraticTrustSolver",
    "QuarticSolveResult",
    "QuarticSolver",
    "SubspaceQuartic",
]
