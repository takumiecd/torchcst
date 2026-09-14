"""Reference fixed-point solver for candidate-dependent normalized updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from .base import NormalizedSolver


@runtime_checkable
class NormalizedEvaluation(Protocol):
    """Candidate-dependent tensor evaluation used by normalized solvers."""

    @property
    def point_shape(self) -> tuple[int, int]: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def at(self, displacement: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class NormalizedSolveResult:
    """Diagnostics from a normalized fixed-point solve."""

    displacement: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool
    on_boundary: bool
    solver_mode: Literal["fixed_point", "zero_point"] = "fixed_point"


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
        numerator: NormalizedEvaluation,
        denominator: NormalizedEvaluation,
        *,
        learning_rate: float,
    ) -> None:
        if not isinstance(numerator, NormalizedEvaluation):
            raise TypeError("numerator must implement NormalizedEvaluation")
        if not isinstance(denominator, NormalizedEvaluation):
            raise TypeError("denominator must implement NormalizedEvaluation")
        if numerator.point_shape != denominator.point_shape:
            raise ValueError("numerator and denominator shapes must match")
        if (
            numerator.device != denominator.device
            or numerator.dtype != denominator.dtype
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
        return self.numerator.point_shape

    @property
    def device(self) -> torch.device:
        return self.numerator.device

    @property
    def dtype(self) -> torch.dtype:
        return self.numerator.dtype

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

    @staticmethod
    def _at_zero(evaluation: NormalizedEvaluation) -> Tensor:
        """Evaluate a normalized component at zero without its displacement terms."""

        at_zero = getattr(evaluation, "at_zero", None)
        if at_zero is not None:
            return at_zero()
        zero = torch.zeros(
            evaluation.point_shape,
            device=evaluation.device,
            dtype=evaluation.dtype,
        )
        return evaluation.at(zero)

    def normalized_gradient_at_zero(self) -> Tensor:
        """Return ``N(0) / D(0)`` using the zero-point fast path when available."""

        numerator = self._at_zero(self.numerator)
        denominator = self._at_zero(self.denominator)
        finite = torch.isfinite(numerator).all() & torch.isfinite(denominator).all()
        if not bool(finite):
            raise FloatingPointError("normalized update became non-finite")
        if not bool((denominator > 0).all()):
            raise FloatingPointError("normalized update denominator is not positive")
        return numerator / denominator

    def fixed_point(self, displacement: Tensor) -> Tensor:
        """Evaluate ``-learning_rate * N(d) / D(d)``."""

        return -self.learning_rate * self.normalized_gradient(displacement)

    def fixed_point_at_zero(self) -> Tensor:
        """Evaluate ``-learning_rate * N(0) / D(0)`` without curvature contractions."""

        return -self.learning_rate * self.normalized_gradient_at_zero()


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
                device=problem.device,
                dtype=problem.dtype,
            )
            iterations = 0
            for iteration in range(self.max_iter):
                target = _project_ball(
                    (
                        problem.fixed_point_at_zero()
                        if iteration == 0
                        else problem.fixed_point(displacement)
                    ),
                    trust_radius,
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


class NormalizedBoxFixedPointSolver(NormalizedFixedPointSolver):
    """Damped fixed-point iteration with an elementwise trust box.

    ``trust_radius`` is retained in the solver contract for compatibility,
    but is interpreted here as the half-width of the box:

    ``-trust_radius <= d[k, p] <= trust_radius``.

    Unlike :class:`NormalizedFixedPointSolver`, the bound is independent of
    the number of atoms and parameter coordinates.
    """

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
                device=problem.device,
                dtype=problem.dtype,
            )
            iterations = 0
            for iteration in range(self.max_iter):
                target = _project_box(
                    (
                        problem.fixed_point_at_zero()
                        if iteration == 0
                        else problem.fixed_point(displacement)
                    ),
                    trust_radius,
                )
                candidate = _project_box(
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

            target = _project_box(problem.fixed_point(displacement), trust_radius)
            residual = torch.linalg.vector_norm(target - displacement)
            scale = torch.maximum(
                torch.ones((), device=residual.device, dtype=residual.dtype),
                torch.linalg.vector_norm(displacement),
            )
            converged = bool(residual <= self.tolerance * scale)
            on_boundary = bool(
                torch.any(
                    torch.abs(displacement) >= trust_radius * (1.0 - 1e-6)
                )
            )
            return NormalizedSolveResult(
                displacement=displacement.detach(),
                residual_norm=residual.detach(),
                iterations=iterations,
                converged=converged,
                on_boundary=on_boundary,
            )

    def project_displacement(
        self,
        value: Tensor,
        *,
        trust_radius: float,
    ) -> Tensor:
        if trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")
        return _project_box(value, trust_radius)

    def displacement_is_valid(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        return bool(
            torch.abs(displacement).amax()
            <= trust_radius * (1.0 + tolerance)
        )

    def displacement_is_on_boundary(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        return bool(
            torch.any(
                torch.abs(displacement) >= trust_radius * (1.0 - 1e-6)
            )
        )


def _project_ball(value: Tensor, radius: float) -> Tensor:
    norm = torch.linalg.vector_norm(value)
    return value * (radius / norm.clamp_min(1e-30)).clamp(max=1.0)


def _project_box(value: Tensor, radius: float) -> Tensor:
    return value.clamp(min=-radius, max=radius)
