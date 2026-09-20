"""Shared N/D evaluation, trust constraints, and iterative displacement solvers.

The Normalized and Quadratic families use the same ingredients:

- ``N(d)`` and ``D(d)`` evaluate a candidate-dependent vector
- ``-η N(d)/D(d)`` is that vector scaled by the learning rate
- a trust constraint (global ball or coordinate box) projects proposals

They differ only in how the scaled vector is applied:

- replace: ``d ← Π(-η N/D)``
- accumulate: ``d ← Π(d - η N/D)``
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from .base import NormalizedSolver


@runtime_checkable
class NormalizedEvaluation(Protocol):
    """Candidate-dependent tensor evaluation used by N/D solvers."""

    @property
    def point_shape(self) -> tuple[int, int]: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def at(self, displacement: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class NormalizedSolveResult:
    """Diagnostics from an N/D displacement solve."""

    displacement: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool
    on_boundary: bool
    solver_mode: Literal["fixed_point", "zero_point", "quadratic", "gradient"] = (
        "fixed_point"
    )


class NormalizedUpdateProblem:
    """Candidate-dependent ``N(d)`` and ``D(d)`` evaluations.

    This object does not choose an update rule.  It only evaluates the
    scaled vector ``-η N(d)/D(d)``.  Solvers decide whether that vector
    replaces ``d`` or is added to it.
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
        return _ratio(numerator, denominator)

    @staticmethod
    def _at_zero(evaluation: NormalizedEvaluation) -> Tensor:
        """Evaluate a component at zero without its displacement terms."""

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

        return _ratio(
            self._at_zero(self.numerator),
            self._at_zero(self.denominator),
        )

    def scaled_step(self, displacement: Tensor) -> Tensor:
        """Return ``-learning_rate * N(d) / D(d)``."""

        return -self.learning_rate * self.normalized_gradient(displacement)

    def scaled_step_at_zero(self) -> Tensor:
        """Return ``-learning_rate * N(0) / D(0)`` without curvature contractions."""

        return -self.learning_rate * self.normalized_gradient_at_zero()

    def fixed_point(self, displacement: Tensor) -> Tensor:
        """Historical name for :meth:`scaled_step`."""

        return self.scaled_step(displacement)

    def fixed_point_at_zero(self) -> Tensor:
        """Historical name for :meth:`scaled_step_at_zero`."""

        return self.scaled_step_at_zero()


def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    finite = torch.isfinite(numerator).all() & torch.isfinite(denominator).all()
    if not bool(finite):
        raise FloatingPointError("normalized update became non-finite")
    if not bool((denominator > 0).all()):
        raise FloatingPointError("normalized update denominator is not positive")
    return numerator / denominator


class _TrustConstraint(ABC):
    """Feasible-set geometry for a candidate displacement."""

    @abstractmethod
    def project(self, value: Tensor, radius: float) -> Tensor: ...

    @abstractmethod
    def is_valid(self, displacement: Tensor, radius: float) -> bool: ...

    @abstractmethod
    def on_boundary(self, displacement: Tensor, radius: float) -> bool: ...


class _GlobalBallConstraint(_TrustConstraint):
    """One Euclidean ball over the whole ``[K, P]`` table."""

    def project(self, value: Tensor, radius: float) -> Tensor:
        _require_positive_radius(radius)
        norm = torch.linalg.vector_norm(value)
        return value * (radius / norm.clamp_min(1e-30)).clamp(max=1.0)

    def is_valid(self, displacement: Tensor, radius: float) -> bool:
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        return bool(
            torch.linalg.vector_norm(displacement)
            <= radius * (1.0 + tolerance)
        )

    def on_boundary(self, displacement: Tensor, radius: float) -> bool:
        return bool(
            torch.linalg.vector_norm(displacement) >= radius * (1.0 - 1e-6)
        )


class _CoordinateBoxConstraint(_TrustConstraint):
    """Elementwise box ``-radius <= d[k, p] <= radius``."""

    def project(self, value: Tensor, radius: float) -> Tensor:
        _require_positive_radius(radius)
        return value.clamp(min=-radius, max=radius)

    def is_valid(self, displacement: Tensor, radius: float) -> bool:
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        return bool(
            torch.abs(displacement).amax() <= radius * (1.0 + tolerance)
        )

    def on_boundary(self, displacement: Tensor, radius: float) -> bool:
        return bool(
            torch.any(torch.abs(displacement) >= radius * (1.0 - 1e-6))
        )


def _require_positive_radius(radius: float) -> None:
    if radius <= 0.0:
        raise ValueError("trust_radius must be positive")


class _IterativeNDSolver(NormalizedSolver):
    """Shared damped loop over an N/D scaled step and a trust constraint."""

    solver_mode: Literal["fixed_point", "gradient"] = "fixed_point"
    _accumulate = False
    _constraint_type: type[_TrustConstraint] = _GlobalBallConstraint

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
        self._constraint = self._constraint_type()

    def project_displacement(
        self,
        value: Tensor,
        *,
        trust_radius: float,
    ) -> Tensor:
        return self._constraint.project(value, trust_radius)

    def displacement_is_valid(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        return self._constraint.is_valid(displacement, trust_radius)

    def displacement_is_on_boundary(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        return self._constraint.on_boundary(displacement, trust_radius)

    def _apply_step(
        self,
        displacement: Tensor,
        step: Tensor,
        *,
        trust_radius: float,
    ) -> Tensor:
        proposed = displacement + step if self._accumulate else step
        return self.project_displacement(proposed, trust_radius=trust_radius)

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
                step = (
                    problem.scaled_step_at_zero()
                    if iteration == 0
                    else problem.scaled_step(displacement)
                )
                target = self._apply_step(
                    displacement,
                    step,
                    trust_radius=trust_radius,
                )
                if self.damping == 1.0:
                    candidate = target
                else:
                    candidate = self.project_displacement(
                        (1.0 - self.damping) * displacement
                        + self.damping * target,
                        trust_radius=trust_radius,
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

            step = problem.scaled_step(displacement)
            target = self._apply_step(
                displacement,
                step,
                trust_radius=trust_radius,
            )
            residual = torch.linalg.vector_norm(target - displacement)
            scale = torch.maximum(
                torch.ones((), device=residual.device, dtype=residual.dtype),
                torch.linalg.vector_norm(displacement),
            )
            return NormalizedSolveResult(
                displacement=displacement.detach(),
                residual_norm=residual.detach(),
                iterations=iterations,
                converged=bool(residual <= self.tolerance * scale),
                on_boundary=self.displacement_is_on_boundary(
                    displacement,
                    trust_radius=trust_radius,
                ),
                solver_mode=self.solver_mode,
            )


class NormalizedFixedPointSolver(_IterativeNDSolver):
    """Replace ``d`` with ``Π(-η N/D)`` on the global Euclidean ball."""


class NormalizedBoxFixedPointSolver(_IterativeNDSolver):
    """Replace ``d`` with ``Π(-η N/D)`` inside an elementwise box."""

    _constraint_type = _CoordinateBoxConstraint


class QuadraticGradientSolver(_IterativeNDSolver):
    """Add ``-η N/D`` to ``d`` on the global Euclidean ball."""

    update_rule = "quadratic"
    solver_mode = "gradient"
    _accumulate = True


class QuadraticBoxGradientSolver(_IterativeNDSolver):
    """Add ``-η N/D`` to ``d`` inside an elementwise box."""

    update_rule = "quadratic"
    solver_mode = "gradient"
    _accumulate = True
    _constraint_type = _CoordinateBoxConstraint
