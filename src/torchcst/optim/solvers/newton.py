"""Dense regularized Newton models constrained directly to the displacement ball."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal, Protocol

import torch
from torch import Tensor

from ..problem import QuarticProblem
from .base import QuarticSolver
from .quartic import FullQuartic, QuarticSolveResult


class _NewtonModel(Protocol):
    def value_and_gradient(self, displacement: Tensor) -> tuple[Tensor, Tensor]: ...
    def hessian(self, displacement: Tensor) -> Tensor: ...


def _spectral_ball_minimum(
    eigenvalues: Tensor, eigenvectors: Tensor, rhs: Tensor, radius: float
) -> Tensor:
    """Globally minimize .5 z.T Q z - rhs.T z on a Euclidean ball.

    Q's eigendecomposition is supplied by the caller. Solve the scalar secular
    equation in host doubles after one small transfer, avoiding dozens of tiny
    GPU launches/synchronizations. Handle singular PSD interiors and the
    indefinite hard case explicitly. The matrix/eigenvectors stay on device.
    """

    spectral_rhs = eigenvectors.T @ rhs
    values, coefficients = (
        torch.stack((eigenvalues, spectral_rhs)).detach().cpu().tolist()
    )
    scale = max(max(abs(x) for x in values), 1e-30)
    tolerance = 8.0 * torch.finfo(eigenvalues.dtype).eps * scale
    rhs_norm = math.sqrt(sum(x * x for x in coefficients))
    lower = max(0.0, -values[0])
    shifted = [x + lower for x in values]
    null = [i for i, x in enumerate(shifted) if x <= tolerance]
    null_set = set(null)
    endpoint = [
        0.0 if i in null_set else b / shifted[i] for i, b in enumerate(coefficients)
    ]
    endpoint_square = sum(x * x for x in endpoint)
    small_null_rhs = all(
        abs(coefficients[i])
        <= 8.0 * torch.finfo(eigenvalues.dtype).eps * max(rhs_norm, 1e-30)
        for i in null
    )
    if small_null_rhs and endpoint_square <= radius * radius:
        if lower > 0.0:
            # Complete the minimum-norm solution in a leftmost eigendirection.
            endpoint[0] = math.copysign(
                math.sqrt(max(0.0, radius * radius - endpoint_square)), coefficients[0]
            )
        return eigenvectors @ eigenvalues.new_tensor(endpoint)

    upper = lower + rhs_norm / radius
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        norm = math.hypot(
            *(b / max(v + midpoint, 1e-300) for v, b in zip(values, coefficients))
        )
        if norm > radius:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= 2e-14 * max(scale, abs(upper)):
            break
    solution = eigenvalues.new_tensor(
        [b / max(v + upper, 1e-300) for v, b in zip(values, coefficients)]
    )
    result = eigenvectors @ solution
    # Eigensolver roundoff can put the reconstructed point just outside the ball.
    return result * torch.clamp(
        radius
        / torch.linalg.vector_norm(result).clamp_min(torch.finfo(result.dtype).tiny),
        max=1.0,
    )


class BallNewton(QuarticSolver):
    """Exact quartic derivatives, direct ball constraint, adaptive regularization.

    Quadratic subproblems retain negative curvature. Every accepted update
    decreases the original quartic. Convergence reports first-order constrained
    stationarity; an interior stationary saddle is not accepted as convergence.
    """

    def __init__(
        self,
        *,
        starts: int = 1,
        max_iter: int = 30,
        max_evaluations: int = 100,
        tolerance_grad: float = 1e-5,
        relative_tolerance_grad: float = 0.0,
        acceptance_ratio: float = 0.1,
        execution: Literal["eager", "compiled"] = "eager",
        secular_solver: Literal["host", "device"] = "host",
    ) -> None:
        for name, value in (
            ("starts", starts),
            ("max_iter", max_iter),
            ("max_evaluations", max_evaluations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(tolerance_grad) or tolerance_grad <= 0:
            raise ValueError("tolerance_grad must be finite and positive")
        if not math.isfinite(relative_tolerance_grad) or relative_tolerance_grad < 0:
            raise ValueError("relative_tolerance_grad must be finite and nonnegative")
        if not 0 < acceptance_ratio < 1:
            raise ValueError("acceptance_ratio must lie between zero and one")
        if execution not in ("eager", "compiled"):
            raise ValueError("execution must be eager or compiled")
        if secular_solver not in ("host", "device"):
            raise ValueError("secular_solver must be host or device")
        self.execution = execution
        self.secular_solver = secular_solver
        self.max_iter = max_iter
        self.starts = starts
        self.max_evaluations = max_evaluations
        self.tolerance_grad = tolerance_grad
        self.relative_tolerance_grad = relative_tolerance_grad
        self.acceptance_ratio = acceptance_ratio

    def solve(
        self, problem: QuarticProblem, *, trust_radius: float
    ) -> QuarticSolveResult:
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if not math.isfinite(trust_radius) or trust_radius <= 0:
            raise ValueError("trust_radius must be finite and positive")
        initial_points = FullQuartic(starts=self.starts)._initial_points(
            problem, trust_radius
        )
        model = problem
        if self.execution == "compiled":
            from ._compiled import CompiledQuarticModel

            model = CompiledQuarticModel(problem)
        results = [
            self._solve(model, initial, trust_radius) for initial in initial_points
        ]
        index = (
            0
            if len(results) == 1
            else min(range(len(results)), key=lambda i: float(results[i].objective))
        )
        return replace(
            results[index],
            start_index=index,
            iterations=sum(result.iterations for result in results),
            evaluations=sum(result.evaluations for result in results)
            + int(self.starts > 1),
        )

    def _solve(
        self, problem: _NewtonModel, initial: Tensor, radius: float
    ) -> QuarticSolveResult:
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("trust_radius must be finite and positive")
        if initial.dtype not in (torch.float32, torch.float64):
            raise ValueError("BallNewton requires float32 or float64")
        device_control = (
            self.execution == "compiled" and self.secular_solver == "device"
        )
        if device_control:
            from ._compiled import call_compiled

        with torch.no_grad():
            current = self._project(initial.detach().clone(), radius)
            value, gradient = problem.value_and_gradient(current)
            if not bool(torch.isfinite(value) & torch.isfinite(gradient).all()):
                raise RuntimeError(
                    "quartic objective is non-finite at the initial point"
                )
            evaluations, iterations = 1, 0
            if device_control:
                initial_norm = self._projected_norm(current, gradient, radius).double()
                threshold = (self.relative_tolerance_grad * initial_norm).clamp_min(
                    self.tolerance_grad
                )
                regularization = current.new_zeros((), dtype=torch.float64)
            else:
                initial_norm = float(self._projected_norm(current, gradient, radius))
                threshold = max(
                    self.tolerance_grad, self.relative_tolerance_grad * initial_norm
                )
                regularization = 0.0
            converged = False
            while iterations < self.max_iter and evaluations < self.max_evaluations:
                matrix = problem.hessian(current).detach()
                if not torch.isfinite(matrix).all():
                    break
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
                if device_control:
                    curvature_scale, stop = call_compiled(
                        "newton_status",
                        eigenvalues,
                        current,
                        gradient,
                        threshold,
                        radius,
                    )
                    stop = bool(stop)
                else:
                    curvature_scale = max(float(eigenvalues.abs().max()), 1e-12)
                    projected_norm = float(
                        self._projected_norm(current, gradient, radius)
                    )
                    interior = float(torch.linalg.vector_norm(current)) < radius * (
                        1 - 1e-5
                    )
                    negative_curvature = (
                        float(eigenvalues[0])
                        < -8 * torch.finfo(current.dtype).eps * curvature_scale
                    )
                    stop = projected_norm <= threshold and not (
                        interior and negative_curvature
                    )
                if stop:
                    converged = True
                    break
                accepted = False
                # Reuse the Hessian/eigenvectors while increasing regularization.
                for _ in range(12):
                    if evaluations >= self.max_evaluations:
                        break
                    shifted_values = eigenvalues + regularization
                    rhs = (
                        matrix @ current.reshape(-1)
                        - gradient.reshape(-1)
                        + regularization * current.reshape(-1)
                    )
                    candidate = self._spectral_minimum(
                        shifted_values, eigenvectors, rhs, radius
                    ).reshape_as(current)
                    direction = candidate - current
                    # Feasible chords also let us leave a stationary saddle:
                    # damping alone can jump from a boundary minimizer to zero.
                    for backtrack in range(4):
                        if evaluations >= self.max_evaluations:
                            break
                        candidate = current + (0.5**backtrack) * direction
                        step = (candidate - current).reshape(-1)
                        candidate_value, candidate_gradient = (
                            problem.value_and_gradient(candidate)
                        )
                        evaluations += 1
                        if device_control:
                            accept = call_compiled(
                                "accept_candidate",
                                gradient,
                                step,
                                matrix,
                                regularization,
                                value,
                                candidate_value,
                                candidate_gradient,
                                self.acceptance_ratio,
                            )
                        else:
                            predicted = (
                                -torch.dot(gradient.reshape(-1), step)
                                - 0.5 * torch.dot(step, matrix @ step)
                                - 0.5 * regularization * torch.dot(step, step)
                            )
                            actual = value - candidate_value
                            accept = (
                                torch.isfinite(candidate_value)
                                & torch.isfinite(candidate_gradient).all()
                                & (predicted > 0)
                                & (actual > 0)
                                & (actual >= self.acceptance_ratio * predicted)
                            )
                        if bool(accept):
                            current, value, gradient = (
                                candidate,
                                candidate_value,
                                candidate_gradient,
                            )
                            regularization *= 0.25
                            iterations += 1
                            accepted = True
                            break
                    if accepted:
                        break
                    if device_control:
                        regularization = torch.maximum(
                            regularization * 4, 0.01 * curvature_scale
                        )
                    else:
                        regularization = max(regularization * 4, 0.01 * curvature_scale)
                if not accepted:
                    break
            projected_norm = self._projected_norm(current, gradient, radius)
            # A small first-order residual after the final accepted step has not
            # yet passed the interior curvature check; do not overclaim convergence.
            boundary_tolerance = max(
                10 * torch.finfo(current.dtype).eps * radius, 1e-4 * radius
            )
            return QuarticSolveResult(
                displacement=current.detach().clone(),
                objective=value.detach().clone(),
                projected_gradient_norm=projected_norm.detach().clone(),
                start_index=0,
                iterations=iterations,
                evaluations=evaluations,
                converged=converged,
                on_boundary=bool(
                    torch.linalg.vector_norm(current) >= radius - boundary_tolerance
                ),
            )

    def _spectral_minimum(self, eigenvalues, eigenvectors, rhs, radius):
        if self.secular_solver == "host":
            return _spectral_ball_minimum(eigenvalues, eigenvectors, rhs, radius)
        from ._compiled import call_compiled, spectral_ball_device

        if self.execution == "compiled":
            return call_compiled(
                "spectral_ball_device", eigenvalues, eigenvectors, rhs, radius
            )
        return spectral_ball_device(eigenvalues, eigenvectors, rhs, radius)

    @staticmethod
    def _project(value: Tensor, radius: float) -> Tensor:
        return value * torch.clamp(
            radius
            / torch.linalg.vector_norm(value).clamp_min(torch.finfo(value.dtype).tiny),
            max=1.0,
        )

    @classmethod
    def _projected_norm(cls, value: Tensor, gradient: Tensor, radius: float) -> Tensor:
        return torch.linalg.vector_norm(value - cls._project(value - gradient, radius))
