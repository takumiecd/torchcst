"""Contract for replaceable quartic-problem solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..problem import QuarticProblem

if TYPE_CHECKING:
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
