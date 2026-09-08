"""Reorthogonalized symmetric Krylov trust solve with bounded retained bases.

GLTR-style subspace restriction, with an explicitly projected small matrix
instead of relying on an exact finite-precision three-term recurrence. The
original Euclidean ball is preserved; no coordinate metric is substituted.
"""

from dataclasses import dataclass

import torch

from torchcst._derivatives.block_eigh import block_eigh
from torchcst._runtime.validation import require

from ._trust_pcg import actual_certificate, rounded
from .solvers.quartic import QuarticSolveResult


@dataclass(frozen=True)
class KrylovResult(QuarticSolveResult):
    shift: torch.Tensor
    relative_residual: torch.Tensor
    relative_complementarity: torch.Tensor
    basis_dimension: torch.Tensor
    basis_capacity: int
    basis_bytes: int


def orthogonalize(vector, basis):
    # Twice-applied classical Gram-Schmidt, FP64. Unused basis rows are zero.
    residual = vector - basis.T @ (basis @ vector)
    return residual - basis.T @ (basis @ residual)


def restricted_solve(matrix, rhs, radius):
    finite = torch.isfinite(matrix).all() & torch.isfinite(rhs).all()
    matrix = torch.where(finite, matrix, 0)
    rhs = torch.where(finite, rhs, 0)
    values, vectors = block_eigh(matrix[None])
    values, vectors = values[0], vectors[0]
    pad = vectors.shape[-1] - rhs.numel()
    padded = torch.nn.functional.pad(matrix, (0, pad, 0, pad))
    eigen_error = (padded @ vectors - vectors * values).norm()
    orth_error = (
        vectors.T @ vectors
        - torch.eye(values.numel(), device=rhs.device, dtype=rhs.dtype)
    ).norm()
    valid = (
        finite
        & torch.isfinite(eigen_error)
        & (eigen_error <= 1e-9 * matrix.norm().clamp_min(1e-30))
        & (orth_error <= 1e-9 * values.numel())
        & (values.min() >= -1e-12 * matrix.norm().clamp_min(1e-30))
    )
    coeff = vectors.T @ torch.nn.functional.pad(rhs, (0, pad))
    if rhs.is_cuda:
        from torchcst._derivatives.tangent_triton import spectral_solution

        y, shift = spectral_solution(values.clamp_min(0), coeff, radius)
    else:
        from ._tangent_device import spectral_ball

        y, shift = spectral_ball(values.clamp_min(0), coeff, radius)
    return (vectors @ y)[: rhs.numel()], shift, valid


@torch.no_grad()
def solve(
    problem,
    *,
    radius,
    basis_size=128,
    basis_memory_mb=64.0,
    check_interval=32,
    rtol=1e-5,
):
    rhs = -problem.linear.double().contiguous()
    flat = rhs.flatten()
    n = flat.numel()
    # This cap covers the two persistent FP64 N-by-m arrays. Reduced matrices,
    # kernel factors, transient orthogonalization and graph caches are separate.
    capacity = min(n, basis_size, int(basis_memory_mb * 2**20) // (16 * n))
    if capacity < 1:
        raise ValueError("Krylov basis budget cannot hold one vector and its H image")
    basis = flat.new_zeros(capacity, n)
    images = torch.zeros_like(basis)
    norm = flat.norm()
    healthy = torch.isfinite(flat).all()
    done = healthy & (norm == 0)
    q = torch.where(healthy & (norm > 0), flat / norm.clamp_min(1e-300), 0)
    enabled = healthy & (norm > 0)
    dimension = torch.zeros((), device=rhs.device, dtype=torch.int32)
    evaluations = torch.zeros_like(dimension)
    best = torch.zeros_like(problem.linear)
    best_shift = flat.new_zeros(())
    for k in range(capacity):
        if not rhs.is_cuda and bool(done):
            break
        active = enabled & ~done
        q = torch.where(active, q, 0)
        aq = problem.operator(q.reshape_as(rhs), active).flatten()
        healthy = healthy & torch.isfinite(aq).all()
        aq = torch.where(healthy & active, aq, 0)
        basis[k].copy_(q)
        images[k].copy_(aq)
        dimension = dimension + active.to(dimension.dtype)
        evaluations = evaluations + active.to(evaluations.dtype)
        residual = orthogonalize(aq, basis)
        beta = residual.norm()
        enabled = (
            active
            & healthy
            & (beta > 64 * torch.finfo(rhs.dtype).eps * aq.norm().clamp_min(1e-300))
        )
        q = torch.where(enabled, residual / beta.clamp_min(1e-300), 0)
        size = k + 1
        if size % check_interval and size != capacity:
            continue
        current, applied = basis[:size], images[:size]
        projected = current @ applied.T
        projected = 0.5 * (projected + projected.T)
        y, shift, small_ok = restricted_solve(projected, current @ flat, radius)
        candidate = rounded(
            (current.T @ y).reshape_as(rhs), problem.linear.dtype, radius
        )
        hd = problem.operator(candidate.double(), ~done)
        relative, comp, valid = actual_certificate(
            candidate, hd, rhs, shift, radius, rtol
        )
        gram = current @ current.T
        nonzero = current.square().sum(-1) > 0
        orth_ok = (gram - torch.diag(nonzero.to(gram.dtype))).norm() <= 1e-9 * size
        accept = ~done & healthy & small_ok & orth_ok & valid
        best = torch.where(accept, candidate, best)
        best_shift = torch.where(accept, shift, best_shift)
        evaluations = evaluations + (~done).to(evaluations.dtype)
        done = done | accept
    hd = problem.operator(best.double())
    relative, comp, valid = actual_certificate(best, hd, rhs, best_shift, radius, rtol)
    valid = valid & done & healthy
    require(
        valid,
        "Krylov tangent solver exhausted its basis or failed KKT validation",
        FloatingPointError,
    )
    return KrylovResult(
        best,
        ((problem.linear.double() * best).sum() + 0.5 * (best * hd).sum()).detach(),
        (hd + best_shift * best.double() - rhs).norm(),
        0,
        dimension,
        evaluations + 1,
        valid,
        best.double().norm() >= radius * (1 - rtol),
        best_shift,
        relative,
        comp,
        dimension,
        capacity,
        16 * n * capacity,
    )
