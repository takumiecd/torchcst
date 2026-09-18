"""Trust-region minimum of the local Taylor model ``⟨g, d⟩ + ½ ⟨d, H d⟩``."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .base import NormalizedSolver
from .nd import (
    NormalizedEvaluation,
    NormalizedSolveResult,
    NormalizedUpdateProblem,
)


class QuadraticTrustSolver(NormalizedSolver):
    """Minimize the local quadratic on a per-atom Euclidean ball.

    The numerator is the Taylor model ``m(d) = ⟨g, d⟩ + ½ ⟨d, H d⟩``.
    Each atom solves that trust-region subproblem.  Negative curvature
    goes to the ball boundary; a positive-definite interior critical
    point is the Newton step.  This is not ``d = -η (g + H d)``.
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

        affine = _numerator_affine(problem.numerator)
        if affine is None:
            start = torch.zeros(
                problem.point_shape,
                device=problem.device,
                dtype=problem.dtype,
            )
            displacement = self.project_displacement(
                problem.fixed_point(start),
                trust_radius=trust_radius,
            )
        else:
            displacement = self.project_displacement(
                _quadratic_atom_ball_minimum(
                    affine[0],
                    affine[1],
                    radius=trust_radius,
                ),
                trust_radius=trust_radius,
            )
        residual = torch.zeros((), device=problem.device, dtype=problem.dtype)
        return NormalizedSolveResult(
            displacement=displacement.detach(),
            residual_norm=residual,
            iterations=1,
            converged=True,
            on_boundary=self.displacement_is_on_boundary(
                displacement,
                trust_radius=trust_radius,
            ),
            solver_mode="quadratic",
        )

    def project_displacement(
        self,
        value: Tensor,
        *,
        trust_radius: float,
    ) -> Tensor:
        if trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")
        return _project_atom_balls(value, trust_radius)

    def displacement_is_valid(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        tolerance = 10.0 * torch.finfo(displacement.dtype).eps
        return bool(
            torch.linalg.vector_norm(displacement, dim=-1).amax()
            <= trust_radius * (1.0 + tolerance)
        )

    def displacement_is_on_boundary(
        self,
        displacement: Tensor,
        *,
        trust_radius: float,
    ) -> bool:
        return bool(
            torch.linalg.vector_norm(displacement, dim=-1).amax()
            >= trust_radius * (1.0 - 1e-6)
        )


def _numerator_affine(
    evaluation: NormalizedEvaluation,
) -> tuple[Tensor, Tensor] | None:
    """Return ``(g, H)`` when the numerator is an atom-local affine map."""

    corrected = getattr(evaluation, "corrected", evaluation)
    constant = getattr(corrected, "constant", None)
    linear = getattr(corrected, "linear", None)
    if not isinstance(constant, Tensor) or not isinstance(linear, Tensor):
        return None
    if constant.ndim != 2:
        return None
    if linear.shape != (*constant.shape, constant.shape[-1]):
        return None
    return constant, linear


def _quadratic_atom_ball_minimum(
    constant: Tensor,
    linear: Tensor,
    *,
    radius: float,
) -> Tensor:
    """Minimize ``⟨g, d⟩ + ½ ⟨d, H d⟩`` on each atom's Euclidean ball."""

    hessian = 0.5 * (linear + linear.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    return _batched_spectral_ball_minimum(
        eigenvalues,
        eigenvectors,
        -constant,
        radius,
    )


def _batched_spectral_ball_minimum(
    eigenvalues: Tensor,
    eigenvectors: Tensor,
    rhs: Tensor,
    radius: float,
) -> Tensor:
    """Batched Moré–Sorensen ball minimum for atom-local quadratics."""

    values = eigenvalues.double()
    coefficients = torch.matmul(
        eigenvectors.transpose(-1, -2).double(),
        rhs.double().unsqueeze(-1),
    ).squeeze(-1)
    scale = values.abs().amax(dim=-1).clamp_min(1e-30)
    eps = torch.finfo(eigenvalues.dtype).eps
    lower = (-values[..., 0]).clamp_min(0)
    shifted = values + lower.unsqueeze(-1)
    null = shifted <= 8 * eps * scale.unsqueeze(-1)
    endpoint = torch.where(null, 0.0, coefficients / shifted.clamp_min(1e-300))
    endpoint_square = endpoint.square().sum(dim=-1)
    rhs_norm = coefficients.square().sum(dim=-1).sqrt()
    rhs_tol = 8 * eps * rhs_norm.clamp_min(1e-30)
    small_null_rhs = (
        (~null) | (coefficients.abs() <= rhs_tol.unsqueeze(-1))
    ).all(dim=-1)
    endpoint_valid = small_null_rhs & (endpoint_square <= radius * radius)
    addition = torch.copysign(
        (radius * radius - endpoint_square).clamp_min(0).sqrt(),
        coefficients[..., 0],
    )
    first = torch.where(lower > 0, addition, endpoint[..., 0])
    endpoint = torch.cat((first.unsqueeze(-1), endpoint[..., 1:]), dim=-1)
    upper = lower + rhs_norm / radius
    active = torch.ones(values.shape[0], device=values.device, dtype=torch.bool)
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        denom = (values + midpoint.unsqueeze(-1)).clamp_min(1e-300)
        norm = (coefficients / denom).norm(dim=-1)
        go_right = norm > radius
        lower = torch.where(active & go_right, midpoint, lower)
        upper = torch.where(active & ~go_right, midpoint, upper)
        active = active & (
            (upper - lower) > 2e-14 * torch.maximum(scale, upper.abs())
        )
    solution = coefficients / (values + upper.unsqueeze(-1)).clamp_min(1e-300)
    result = torch.matmul(
        eigenvectors.double(),
        solution.unsqueeze(-1),
    ).squeeze(-1)
    result = _project_atom_balls(result, radius)
    endpoint_result = torch.matmul(
        eigenvectors.double(),
        endpoint.unsqueeze(-1),
    ).squeeze(-1)
    chosen = torch.where(endpoint_valid.unsqueeze(-1), endpoint_result, result)
    return chosen.to(dtype=eigenvalues.dtype)


def _project_atom_balls(value: Tensor, radius: float) -> Tensor:
    norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * (radius / norms.clamp_min(1e-30)).clamp(max=1.0)
