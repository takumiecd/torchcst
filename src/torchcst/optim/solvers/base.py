"""Contract for replaceable quartic-problem solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..problem import QuarticProblem

if TYPE_CHECKING:
    from .normalized import NormalizedSolveResult, NormalizedUpdateProblem
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
    """Solve a candidate-dependent normalized update without owning state."""

    @abstractmethod
    def solve(
        self,
        problem: NormalizedUpdateProblem,
        *,
        trust_radius: float,
    ) -> NormalizedSolveResult: ...
