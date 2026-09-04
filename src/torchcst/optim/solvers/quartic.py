"""Numerical trust-region solver for complete quartic CST objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..problem import QuarticProblem
from .base import QuarticSolver


@dataclass(frozen=True)
class QuarticSolveResult:
    """Best finite local solution found across all starts."""

    displacement: Tensor
    objective: Tensor
    projected_gradient_norm: Tensor
    start_index: int
    iterations: int
    evaluations: int
    converged: bool
    on_boundary: bool


class FullQuartic(QuarticSolver):
    """Multi-start LBFGS solve of the untruncated trust-region quartic."""

    def __init__(
        self,
        *,
        starts: int = 4,
        max_iter: int = 80,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 20,
        warm_start: bool = True,
    ) -> None:
        for name, value in (
            ("starts", starts),
            ("max_iter", max_iter),
            ("history_size", history_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if tolerance_grad <= 0 or tolerance_change <= 0:
            raise ValueError("solver tolerances must be positive")
        if not isinstance(warm_start, bool):
            raise TypeError("warm_start must be a boolean")
        self.starts = starts
        self.max_iter = max_iter
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.history_size = history_size
        self.warm_start = warm_start

    def solve(
        self,
        problem: QuarticProblem,
        *,
        trust_radius: float,
        initial: Tensor | None = None,
    ) -> QuarticSolveResult:
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        if not self.warm_start:
            initial = None
        if initial is not None:
            if initial.shape != problem.point_shape:
                raise ValueError("initial displacement has the wrong shape")
            point = problem.context.current_point
            if initial.device != point.device or initial.dtype != point.dtype:
                raise ValueError("initial displacement must match the problem")
            if not torch.isfinite(initial).all():
                raise FloatingPointError("initial displacement must be finite")
        radius = float(trust_radius)
        candidates = []
        for start_index, start in enumerate(
            self._initial_points(problem, radius, initial=initial)
        ):
            candidate = self._solve_one(
                problem,
                start,
                radius=radius,
                start_index=start_index,
            )
            if candidate is not None:
                candidates.append(candidate)
                if initial is not None and start_index == 0 and candidate.converged:
                    return candidate
        if not candidates:
            raise RuntimeError("quartic solver did not produce a finite candidate")
        return min(candidates, key=lambda result: float(result.objective))

    def _solve_one(
        self,
        problem: QuarticProblem,
        initial: Tensor,
        *,
        radius: float,
        start_index: int,
    ) -> QuarticSolveResult | None:
        unconstrained = nn.Parameter(self._to_unconstrained(initial, radius))
        optimizer = torch.optim.LBFGS(
            (unconstrained,),
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure() -> Tensor:
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            displacement = self._from_unconstrained(unconstrained, radius)
            objective = problem.value(displacement)
            if not torch.isfinite(objective):
                raise FloatingPointError("quartic objective became non-finite")
            objective.backward()
            evaluations += 1
            return objective

        try:
            optimizer.step(closure)
        except FloatingPointError:
            return None

        with torch.no_grad():
            displacement = self._from_unconstrained(unconstrained, radius).detach()
            objective = problem.value(displacement).detach()
            gradient = problem.gradient(displacement).detach()
            if not torch.isfinite(objective) or not torch.isfinite(gradient).all():
                return None
            projected = self._project_ball(displacement - gradient, radius)
            projected_gradient_norm = torch.linalg.vector_norm(
                displacement - projected
            )
            norm = torch.linalg.vector_norm(displacement)
            boundary_tolerance = max(
                10.0 * torch.finfo(displacement.dtype).eps * radius,
                1e-4 * radius,
            )
            on_boundary = bool(norm >= radius - boundary_tolerance)
            converged = bool(projected_gradient_norm <= self.tolerance_grad)
            state = optimizer.state[unconstrained]
            iterations = int(state.get("n_iter", 0))
        return QuarticSolveResult(
            displacement=displacement.clone(),
            objective=objective.clone(),
            projected_gradient_norm=projected_gradient_norm.clone(),
            start_index=start_index,
            iterations=iterations,
            evaluations=evaluations,
            converged=converged,
            on_boundary=on_boundary,
        )

    def _initial_points(
        self,
        problem: QuarticProblem,
        radius: float,
        *,
        initial: Tensor | None,
    ) -> tuple[Tensor, ...]:
        zero = torch.zeros_like(problem.context.current_point)
        points = []
        if initial is not None:
            points.append(self._project_ball(initial.detach().clone(), radius))
            if self.starts > 1:
                points.append(zero)
        else:
            points.append(zero)
        if self.starts == 1:
            return tuple(points)

        if len(points) < self.starts:
            gradient = problem.gradient(zero).detach()
            gradient_norm = torch.linalg.vector_norm(gradient)
            if gradient_norm > 0:
                points.append(-0.25 * radius * gradient / gradient_norm)
            else:
                points.append(zero.clone())

        flat_indices = torch.arange(
            zero.numel(),
            device=zero.device,
            dtype=zero.dtype,
        )
        for index in range(len(points), self.starts):
            direction = torch.sin(
                (index + 1.0) * (flat_indices + 1.0) * 1.618033988749895
            ).reshape_as(zero)
            direction_norm = torch.linalg.vector_norm(direction)
            if direction_norm == 0:
                direction = torch.ones_like(direction)
                direction_norm = torch.linalg.vector_norm(direction)
            points.append(0.5 * radius * direction / direction_norm)
        return tuple(points)

    @staticmethod
    def _from_unconstrained(value: Tensor, radius: float) -> Tensor:
        return radius * value / (1.0 + value.square().sum()).sqrt()

    @staticmethod
    def _to_unconstrained(value: Tensor, radius: float) -> Tensor:
        norm = torch.linalg.vector_norm(value)
        if norm == 0:
            return value.detach().clone()
        maximum = 0.95 * radius
        bounded = torch.where(norm > maximum, value * (maximum / norm), value)
        denominator = (radius**2 - bounded.square().sum()).clamp_min(
            torch.finfo(value.dtype).tiny
        )
        return (bounded / denominator.sqrt()).detach().clone()

    @staticmethod
    def _project_ball(value: Tensor, radius: float) -> Tensor:
        norm = torch.linalg.vector_norm(value)
        if norm <= radius:
            return value
        return value * (radius / norm)
