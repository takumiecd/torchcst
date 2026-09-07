"""Adaptive small-subspace solves retaining the complete restricted quartic."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..problem import QuarticProblem
from .base import QuarticSolver
from .newton import BallNewton
from .quartic import QuarticSolveResult


class SubspaceQuartic(QuarticSolver):
    """Solve exact restricted quartics and expand with full-space residuals.

    Derivative contractions run on the problem device; the small polynomial
    solves run in CPU float64 to avoid fine-grained GPU launch overhead.
    ``evaluations`` includes both reduced-model and full-problem evaluations.
    """

    def __init__(
        self,
        *,
        max_dimension: int = 8,
        max_models: int = 8,
        inner_max_iter: int = 20,
        inner_max_evaluations: int = 100,
        tolerance_grad: float = 1e-5,
    ) -> None:
        for name, value in (
            ("max_dimension", max_dimension),
            ("max_models", max_models),
            ("inner_max_iter", inner_max_iter),
            ("inner_max_evaluations", inner_max_evaluations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_dimension < 3:
            raise ValueError("max_dimension must be at least three")
        if not math.isfinite(tolerance_grad) or tolerance_grad <= 0:
            raise ValueError("tolerance_grad must be finite and positive")
        self.max_dimension = max_dimension
        self.max_models = max_models
        self.tolerance_grad = tolerance_grad
        self.inner_solver = BallNewton(
            max_iter=inner_max_iter,
            max_evaluations=inner_max_evaluations,
            tolerance_grad=min(1e-8, tolerance_grad * 0.1),
        )

    def solve(
        self, problem: QuarticProblem, *, trust_radius: float
    ) -> QuarticSolveResult:
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if not math.isfinite(trust_radius) or trust_radius <= 0:
            raise ValueError("trust_radius must be finite and positive")
        with torch.no_grad():
            current = torch.zeros_like(problem.context.current_point)
            if current.dtype not in (torch.float32, torch.float64):
                raise ValueError("SubspaceQuartic requires float32 or float64")
            value, gradient = problem.value_and_gradient(current)
            if not bool(torch.isfinite(value) & torch.isfinite(gradient).all()):
                raise RuntimeError("quartic is non-finite at zero")
            evaluations, models = 1, 0
            atoms, parameters = current.shape
            matrix = problem.hessian(current)
            indices = torch.arange(atoms, device=current.device)
            blocks = matrix.reshape(atoms, parameters, atoms, parameters)[
                indices, :, indices, :
            ]
            eigenvalues, eigenvectors = torch.linalg.eigh(blocks)
            floor = (1e-3 * eigenvalues.abs().amax(dim=-1, keepdim=True)).clamp_min(
                1e-8
            )
            inverse = (
                eigenvectors / (eigenvalues.abs() + floor).unsqueeze(-2)
            ) @ eigenvectors.transpose(-1, -2)

            def precondition(vector):
                return torch.einsum(
                    "kpq,kq->kp", inverse, vector.reshape_as(current)
                ).reshape(-1)

            basis = current.new_empty(current.numel(), 0)
            for direction in (-gradient.reshape(-1), -precondition(gradient)):
                basis = self._append(basis, direction)
            if basis.shape[1] == 0:
                values, vectors = torch.linalg.eigh(matrix)
                if values[0] < 0:
                    basis = vectors[:, :1]

            for _ in range(self.max_models):
                if basis.shape[1] == 0:
                    break
                restricted = problem.restricted_model(basis)
                initial = (
                    (basis.T @ current.reshape(-1)).detach().cpu().double().unsqueeze(0)
                )
                inner = self.inner_solver._solve(restricted, initial, trust_radius)
                evaluations += inner.evaluations
                candidate = (
                    basis @ inner.displacement.reshape(-1).to(current)
                ).reshape_as(current)
                candidate = BallNewton._project(candidate, trust_radius)
                candidate_value, candidate_gradient = problem.value_and_gradient(
                    candidate
                )
                evaluations += 1
                models += 1
                if bool(
                    torch.isfinite(candidate_value)
                    & torch.isfinite(candidate_gradient).all()
                    & (candidate_value <= value)
                ):
                    current, value, gradient = (
                        candidate,
                        candidate_value,
                        candidate_gradient,
                    )
                residual = current - BallNewton._project(
                    current - gradient, trust_radius
                )
                if torch.linalg.vector_norm(residual) <= self.tolerance_grad:
                    break
                if basis.shape[1] >= min(self.max_dimension, current.numel()):
                    basis = current.new_empty(current.numel(), 0)
                    basis = self._append(basis, current.reshape(-1))
                for direction in (-residual.reshape(-1), -precondition(residual)):
                    if basis.shape[1] < min(self.max_dimension, current.numel()):
                        basis = self._append(basis, direction)

            residual_norm = BallNewton._projected_norm(current, gradient, trust_radius)
            boundary_tolerance = max(
                10 * torch.finfo(current.dtype).eps * trust_radius, 1e-4 * trust_radius
            )
            return QuarticSolveResult(
                displacement=current.detach().clone(),
                objective=value.detach().clone(),
                projected_gradient_norm=residual_norm.detach().clone(),
                start_index=0,
                iterations=models,
                evaluations=evaluations,
                converged=bool(residual_norm <= self.tolerance_grad),
                on_boundary=bool(
                    torch.linalg.vector_norm(current)
                    >= trust_radius - boundary_tolerance
                ),
            )

    @staticmethod
    def _append(basis: Tensor, direction: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(direction)
        if norm <= torch.finfo(direction.dtype).tiny:
            return basis
        candidate = direction / norm
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        remaining = torch.linalg.vector_norm(candidate)
        if remaining <= 32 * torch.finfo(direction.dtype).eps:
            return basis
        return torch.cat((basis, (candidate / remaining).unsqueeze(1)), dim=1)
