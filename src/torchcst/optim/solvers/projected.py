"""Projected limited-memory solver for structured quartic CST objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..problem import QuarticProblem
from .base import QuarticSolver
from .quartic import QuarticSolveResult


@dataclass(frozen=True)
class _Candidate:
    displacement: Tensor
    objective: Tensor
    gradient: Tensor
    projected_gradient: Tensor
    start_index: int


class ProjectedLBFGS(QuarticSolver):
    """Projected L-BFGS with safeguarded starts and a strict evaluation budget."""

    def __init__(
        self,
        *,
        max_iter: int = 20,
        max_evaluations: int = 24,
        history_size: int = 10,
        line_search_steps: int = 8,
        tolerance_grad: float = 1e-6,
        relative_tolerance_grad: float = 1e-3,
        tolerance_change: float = 1e-9,
        backtrack_factor: float = 0.5,
        armijo: float = 1e-4,
    ) -> None:
        for name, value in (
            ("max_iter", max_iter),
            ("max_evaluations", max_evaluations),
            ("history_size", history_size),
            ("line_search_steps", line_search_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if tolerance_grad <= 0 or tolerance_change <= 0:
            raise ValueError("solver tolerances must be positive")
        if relative_tolerance_grad < 0:
            raise ValueError("relative_tolerance_grad must be nonnegative")
        if not 0.0 < backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must satisfy 0 < value < 1")
        if not 0.0 < armijo < 1.0:
            raise ValueError("armijo must satisfy 0 < value < 1")
        self.max_iter = max_iter
        self.max_evaluations = max_evaluations
        self.history_size = history_size
        self.line_search_steps = line_search_steps
        self.tolerance_grad = float(tolerance_grad)
        self.relative_tolerance_grad = float(relative_tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.backtrack_factor = float(backtrack_factor)
        self.armijo = float(armijo)

    def solve(
        self,
        problem: QuarticProblem,
        *,
        trust_radius: float,
    ) -> QuarticSolveResult:
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        radius = float(trust_radius)
        zero = torch.zeros_like(problem.context.current_point)
        zero_candidate = self._evaluate(problem, zero, radius, 0)
        evaluations = 1
        if zero_candidate is None:
            raise RuntimeError("quartic objective is non-finite at zero")

        reference_norm = torch.linalg.vector_norm(
            zero_candidate.projected_gradient
        )
        threshold = max(
            self.tolerance_grad,
            self.relative_tolerance_grad * float(reference_norm),
        )
        if reference_norm <= threshold:
            return self._result(
                zero_candidate,
                iterations=0,
                evaluations=evaluations,
                threshold=threshold,
                radius=radius,
            )

        candidates = [zero_candidate]
        if evaluations < self.max_evaluations:
            gradient_norm = torch.linalg.vector_norm(zero_candidate.gradient)
            if gradient_norm > 0:
                cauchy = -radius * zero_candidate.gradient / gradient_norm
                cauchy_candidate = self._evaluate(
                    problem,
                    cauchy,
                    radius,
                    1,
                )
                evaluations += 1
                if cauchy_candidate is not None:
                    candidates.append(cauchy_candidate)

        current = min(candidates, key=lambda candidate: float(candidate.objective))
        best = current
        history: list[tuple[Tensor, Tensor, Tensor]] = []
        iterations = 0

        while iterations < self.max_iter and evaluations < self.max_evaluations:
            projected_norm = torch.linalg.vector_norm(current.projected_gradient)
            if projected_norm <= threshold:
                break

            direction = self._lbfgs_direction(current.projected_gradient, history)
            fallback = bool(
                (current.projected_gradient * direction).sum() >= 0
            )
            if fallback:
                direction = -current.projected_gradient

            candidate, used = self._line_search(
                problem,
                current,
                direction,
                radius=radius,
                remaining=self.max_evaluations - evaluations,
            )
            evaluations += used
            if candidate is None and not fallback and evaluations < self.max_evaluations:
                candidate, used = self._line_search(
                    problem,
                    current,
                    -current.projected_gradient,
                    radius=radius,
                    remaining=self.max_evaluations - evaluations,
                )
                evaluations += used
            if candidate is None:
                break

            step = candidate.displacement - current.displacement
            objective_change = candidate.objective - current.objective
            projected_change = (
                candidate.projected_gradient - current.projected_gradient
            )
            curvature = (step * projected_change).sum()
            curvature_floor = (
                1e-10
                * torch.linalg.vector_norm(step)
                * torch.linalg.vector_norm(projected_change)
            )
            if curvature > curvature_floor:
                history.append(
                    (
                        step.reshape(-1).detach(),
                        projected_change.reshape(-1).detach(),
                        curvature.detach().reciprocal(),
                    )
                )
                if len(history) > self.history_size:
                    history.pop(0)

            current = candidate
            iterations += 1
            if current.objective < best.objective:
                best = current

            small_step = torch.linalg.vector_norm(step) <= self.tolerance_change * (
                1.0 + torch.linalg.vector_norm(current.displacement)
            )
            small_change = objective_change.abs() <= self.tolerance_change * (
                1.0 + current.objective.abs()
            )
            if bool(small_step and small_change):
                break

        return self._result(
            best,
            iterations=iterations,
            evaluations=evaluations,
            threshold=threshold,
            radius=radius,
        )

    def _line_search(
        self,
        problem: QuarticProblem,
        current: _Candidate,
        direction: Tensor,
        *,
        radius: float,
        remaining: int,
    ) -> tuple[_Candidate | None, int]:
        alpha = 1.0
        used = 0
        for _ in range(min(self.line_search_steps, remaining)):
            displacement = self._project_ball(
                current.displacement + alpha * direction,
                radius,
            )
            step = displacement - current.displacement
            if torch.linalg.vector_norm(step) == 0:
                break
            candidate = self._evaluate(
                problem,
                displacement,
                radius,
                current.start_index,
            )
            used += 1
            if candidate is not None:
                slope = (current.gradient * step).sum()
                if candidate.objective <= current.objective + self.armijo * slope:
                    return candidate, used
            alpha *= self.backtrack_factor
        return None, used

    def _evaluate(
        self,
        problem: QuarticProblem,
        displacement: Tensor,
        radius: float,
        start_index: int,
    ) -> _Candidate | None:
        value, gradient = problem.value_and_gradient(displacement)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            return None
        projected_gradient = displacement - self._project_ball(
            displacement - gradient,
            radius,
        )
        return _Candidate(
            displacement=displacement.detach().clone(),
            objective=value.detach().clone(),
            gradient=gradient.detach().clone(),
            projected_gradient=projected_gradient.detach().clone(),
            start_index=start_index,
        )

    @staticmethod
    def _lbfgs_direction(
        projected_gradient: Tensor,
        history: list[tuple[Tensor, Tensor, Tensor]],
    ) -> Tensor:
        q = projected_gradient.reshape(-1).clone()
        alphas = []
        for step, change, reciprocal_curvature in reversed(history):
            alpha = reciprocal_curvature * torch.dot(step, q)
            alphas.append(alpha)
            q = q - alpha * change
        if history:
            last_step, last_change, _ = history[-1]
            scale = torch.dot(last_step, last_change) / torch.dot(
                last_change, last_change
            ).clamp_min(torch.finfo(q.dtype).tiny)
            q = scale * q
        for (step, change, reciprocal_curvature), alpha in zip(
            history, reversed(alphas)
        ):
            beta = reciprocal_curvature * torch.dot(change, q)
            q = q + step * (alpha - beta)
        return -q.reshape_as(projected_gradient)

    def _result(
        self,
        candidate: _Candidate,
        *,
        iterations: int,
        evaluations: int,
        threshold: float,
        radius: float,
    ) -> QuarticSolveResult:
        projected_norm = torch.linalg.vector_norm(candidate.projected_gradient)
        norm = torch.linalg.vector_norm(candidate.displacement)
        boundary_tolerance = max(
            10.0 * torch.finfo(candidate.displacement.dtype).eps * radius,
            1e-4 * radius,
        )
        return QuarticSolveResult(
            displacement=candidate.displacement.clone(),
            objective=candidate.objective.clone(),
            projected_gradient_norm=projected_norm.clone(),
            start_index=candidate.start_index,
            iterations=iterations,
            evaluations=evaluations,
            converged=bool(projected_norm <= threshold),
            on_boundary=bool(norm >= radius - boundary_tolerance),
        )

    @staticmethod
    def _project_ball(value: Tensor, radius: float) -> Tensor:
        norm = torch.linalg.vector_norm(value)
        if norm <= radius:
            return value
        return value * (radius / norm)
