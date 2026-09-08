"""GPU-resident spectral ball solve; host code only supplies fixed loop bounds."""

import torch

from torchcst._runtime.validation import require

from .solvers.quartic import QuarticSolveResult


def spectral_ball(values, coeff, radius):
    eps = torch.finfo(values.dtype).eps
    cutoff = 8 * eps * values.max().clamp_min(1e-30)
    norm_rhs = coeff.norm()
    visible = values > cutoff
    null_ok = (visible | (coeff.abs() <= 8 * eps * norm_rhs.clamp_min(1e-30))).all()
    interior = torch.where(visible, coeff / torch.where(visible, values, 1), 0)
    boundary = ~null_ok | (interior.norm() > radius)
    low = torch.zeros_like(norm_rhs)
    high = norm_rhs / radius
    for _ in range(80):
        mid = (low + high) * 0.5
        norm = (coeff / (values + mid).clamp_min(1e-300)).norm()
        low, high = (
            torch.where(norm > radius, mid, low),
            torch.where(norm > radius, high, mid),
        )
    shift = torch.where(boundary, high, 0)
    solution = torch.where(
        boundary, coeff / (values + high).clamp_min(1e-300), interior
    )
    return solution, shift


def solve(problem, radius):
    matrix, linear = problem.matrix.double(), problem.linear.double()
    finite = torch.isfinite(matrix).all() & torch.isfinite(linear).all()
    require(finite, "non-finite tangent objective", FloatingPointError)
    matrix = torch.where(finite, matrix, 0)
    linear = torch.where(finite, linear, 0)
    if matrix.is_cuda:
        from torchcst._derivatives.tangent_eigh import eigh

        values, vectors, info = eigh(matrix)
        require(info == 0, "device tangent eigensolve failed", FloatingPointError)
    else:
        values, vectors = torch.linalg.eigh(matrix)
    tolerance = 32 * torch.finfo(problem.matrix.dtype).eps * values.abs().max()
    require(
        (values >= -tolerance).all(),
        "tangent metric is not positive semidefinite",
        FloatingPointError,
    )
    raw_values = values
    values = values.clamp_min(0)
    rhs = -linear.flatten()
    spectral = vectors.T @ rhs
    if matrix.is_cuda:
        from torchcst._derivatives.tangent_triton import spectral_solution

        solution, shift = spectral_solution(values, spectral, radius)
    else:
        solution, shift = spectral_ball(values, spectral, radius)
    d = (vectors @ solution).reshape_as(problem.linear).to(problem.linear)
    d = d * (radius / d.norm().clamp_min(1e-30)).clamp(max=1)
    objective, gradient = problem.value_and_gradient(d)
    stationarity = (gradient + shift * d).norm()
    threshold = 128 * torch.finfo(d.dtype).eps * problem.linear.norm().clamp_min(1)
    # A failed native eigensolve cannot permit a finite but incorrect update.
    eig_error = (matrix @ vectors - vectors * raw_values).norm()
    orth_error = (
        vectors.T @ vectors
        - torch.eye(values.numel(), device=d.device, dtype=values.dtype)
    ).norm()
    require(
        torch.isfinite(eig_error)
        & (eig_error <= 1e-9 * matrix.norm().clamp_min(1))
        & (orth_error <= 1e-9 * values.numel()),
        "device eigensolve residual failed",
        FloatingPointError,
    )
    return QuarticSolveResult(
        d,
        objective,
        stationarity,
        0,
        1,
        1,
        stationarity <= threshold,
        d.norm() >= radius * (1 - 1e-6),
    )
