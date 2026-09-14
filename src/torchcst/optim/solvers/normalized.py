"""Reference fixed-point solver for candidate-dependent normalized updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..moments.denominator import ExpandedDenominatorMoment
from ..moments.numerator import ExpandedNumeratorMoment
from .base import NormalizedSolver


@dataclass(frozen=True)
class NormalizedSolveResult:
    """Diagnostics from a normalized fixed-point solve."""

    displacement: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool
    on_boundary: bool


class NormalizedUpdateProblem:
    """Candidate-dependent normalized update map.

    The fixed-point equation is

    ``d = -learning_rate * N(d) / D(d)``.

    The division is component-wise over the atom table.  This class contains
    only evaluations; iteration and trust-region projection belong to the
    solver below.
    """

    def __init__(
        self,
        numerator: ExpandedNumeratorMoment,
        denominator: ExpandedDenominatorMoment,
        *,
        learning_rate: float,
    ) -> None:
        if not isinstance(numerator, ExpandedNumeratorMoment):
            raise TypeError("numerator must be an ExpandedNumeratorMoment")
        if not isinstance(denominator, ExpandedDenominatorMoment):
            raise TypeError("denominator must be an ExpandedDenominatorMoment")
        if (
            numerator.corrected.constant.shape
            != denominator.corrected.x.shape
        ):
            raise ValueError("numerator and denominator shapes must match")
        if (
            numerator.corrected.constant.device
            != denominator.corrected.x.device
            or numerator.corrected.constant.dtype != denominator.corrected.x.dtype
        ):
            raise ValueError("numerator and denominator must share device and dtype")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (float, int))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive")
        self.numerator = numerator
        self.denominator = denominator
        self.learning_rate = float(learning_rate)

    @property
    def point_shape(self) -> tuple[int, int]:
        return tuple(self.numerator.corrected.constant.shape)

    def normalized_gradient(self, displacement: Tensor) -> Tensor:
        """Return the component-wise ``N(d) / D(d)``."""

        numerator = self.numerator.at(displacement)
        denominator = self.denominator.at(displacement)
        finite = torch.isfinite(numerator).all() & torch.isfinite(denominator).all()
        if not bool(finite):
            raise FloatingPointError("normalized update became non-finite")
        if not bool((denominator > 0).all()):
            raise FloatingPointError("normalized update denominator is not positive")
        return numerator / denominator

    def fixed_point(self, displacement: Tensor) -> Tensor:
        """Evaluate ``-learning_rate * N(d) / D(d)``."""

        return -self.learning_rate * self.normalized_gradient(displacement)


class NormalizedFixedPointSolver(NormalizedSolver):
    """Damped fixed-point iteration with one global Euclidean trust ball."""

    def __init__(
        self,
        *,
        max_iter: int = 32,
        tolerance: float = 1e-7,
        damping: float = 1.0,
    ) -> None:
        if isinstance(max_iter, bool) or not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not math.isfinite(damping) or not 0.0 < damping <= 1.0:
            raise ValueError("damping must be finite and lie in (0, 1]")
        self.max_iter = max_iter
        self.tolerance = float(tolerance)
        self.damping = float(damping)

    def solve(
        self,
        problem: NormalizedUpdateProblem,
        *,
        trust_radius: float,
    ) -> NormalizedSolveResult:
        if not isinstance(problem, NormalizedUpdateProblem):
            raise TypeError("problem must be a NormalizedUpdateProblem")
        if not math.isfinite(trust_radius) or trust_radius <= 0.0:
            raise ValueError("trust_radius must be finite and positive")

        with torch.no_grad():
            displacement = torch.zeros(
                problem.point_shape,
                device=problem.numerator.corrected.constant.device,
                dtype=problem.numerator.corrected.constant.dtype,
            )
            iterations = 0
            for iteration in range(self.max_iter):
                target = _project_ball(
                    problem.fixed_point(displacement), trust_radius
                )
                candidate = _project_ball(
                    (1.0 - self.damping) * displacement
                    + self.damping * target,
                    trust_radius,
                )
                delta = torch.linalg.vector_norm(candidate - displacement)
                displacement = candidate
                iterations = iteration + 1
                scale = torch.maximum(
                    torch.ones((), device=delta.device, dtype=delta.dtype),
                    torch.linalg.vector_norm(displacement),
                )
                if bool(delta <= self.tolerance * scale):
                    break

            target = _project_ball(problem.fixed_point(displacement), trust_radius)
            residual = torch.linalg.vector_norm(target - displacement)
            scale = torch.maximum(
                torch.ones((), device=residual.device, dtype=residual.dtype),
                torch.linalg.vector_norm(displacement),
            )
            converged = bool(residual <= self.tolerance * scale)
            norm = torch.linalg.vector_norm(displacement)
            on_boundary = bool(norm >= trust_radius * (1.0 - 1e-6))
            return NormalizedSolveResult(
                displacement=displacement.detach(),
                residual_norm=residual.detach(),
                iterations=iterations,
                converged=converged,
                on_boundary=on_boundary,
            )


def _project_ball(value: Tensor, radius: float) -> Tensor:
    norm = torch.linalg.vector_norm(value)
    return value * (radius / norm.clamp_min(1e-30)).clamp(max=1.0)
