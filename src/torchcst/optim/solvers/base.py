"""Contract for replaceable quartic-problem solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..problem import QuarticProblem

if TYPE_CHECKING:
    from .nd import NormalizedSolveResult, NormalizedUpdateProblem
    from .quartic import QuarticSolveResult


class QuarticSolver(ABC):
    """Solve a pure local quartic without owning optimizer state."""

    @abstractmethod
    def solve(
        self,
        problem: QuarticProblem,
        *,
        trust_radius: float,
    ) -> QuarticSolveResult: ...


class NormalizedSolver(ABC):
    """Solve a candidate-dependent N/D update without owning state."""

    def project_displacement(
        self,
        value: Tensor,
        *,
        trust_radius: float,
    ) -> Tensor:
        """Project a displacement using the solver's feasible-set geometry."""

        if trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")
        norm = torch.linalg.vector_norm(value)
        return value * (trust_radius / norm.clamp_min(1e-30)).clamp(max=1.0)

    def displacement_is_valid(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        """Check the default global-Euclidean displacement constraint.

        Solvers with a different feasible-set geometry can override this
        method.  Keeping the check on the solver lets the model coordinator
        validate injected solvers without assuming that every constraint is
        an L2 ball.
        """

        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        return bool(
            torch.linalg.vector_norm(displacement)
            <= trust_radius * (1.0 + tolerance)
        )

    def displacement_is_on_boundary(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        return bool(
            torch.linalg.vector_norm(displacement)
            >= trust_radius * (1.0 - 1e-6)
        )

    @abstractmethod
    def solve(
        self,
        problem: NormalizedUpdateProblem,
        *,
        trust_radius: float,
    ) -> NormalizedSolveResult: ...
