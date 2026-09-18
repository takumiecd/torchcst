"""Public N/D solver names.

The evaluation, trust constraints, and iterative loop live in ``nd.py``.
This module re-exports the historical names used by the Normalized and
Quadratic families.
"""

from .nd import (
    NormalizedBoxFixedPointSolver,
    NormalizedEvaluation,
    NormalizedFixedPointSolver,
    NormalizedSolveResult,
    NormalizedUpdateProblem,
    QuadraticBoxGradientSolver,
    QuadraticGradientSolver,
)

__all__ = [
    "NormalizedBoxFixedPointSolver",
    "NormalizedEvaluation",
    "NormalizedFixedPointSolver",
    "NormalizedSolveResult",
    "NormalizedUpdateProblem",
    "QuadraticBoxGradientSolver",
    "QuadraticGradientSolver",
]
