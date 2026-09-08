"""Experimental randomized Nyström preconditioner for damped tangent PCG.

Only the preconditioner is approximated; the solver still applies the full Gram.
The sketch is explicit so callers can reuse it without changing training RNG.
Setup uses FP64 factor actions and O(N*r) storage, never a global Gram matrix.
"""

import torch

from .block_eigh import block_eigh
from .tangent_ops import PreparedFactors


def build(prepared, omega, damping):
    """Return orthonormal U, inverse correction weights, and a device validity flag.

    omega must have orthonormal columns. Implements the shifted Nyström
    construction and scaled inverse of Frangella--Tropp--Udell (2023).
    The explicit sketch rank is bounded by the caller before allocation.
    """
    n, rank = omega.shape
    if n != prepared._point.numel() or not 1 <= rank <= n or damping <= 0:
        raise ValueError("invalid Nyström sketch dimensions or damping")
    p = PreparedFactors(
        prepared._ops,
        prepared._point.double(),
        tuple(
            x.double() for x in (prepared._u, prepared._v, prepared._du, prepared._dv)
        ),
    )
    omega = omega.double()
    y = torch.stack(
        [
            p._device_gram(omega[:, i].reshape_as(p._point)).flatten()
            for i in range(rank)
        ],
        dim=1,
    )
    eye = torch.eye(rank, dtype=y.dtype, device=y.device)
    nu = torch.finfo(y.dtype).eps * y.norm().clamp_min(1e-30)
    shifted = y + nu * omega
    core = omega.T @ shifted
    core = (core + core.T) * 0.5
    c, info = torch.linalg.cholesky_ex(core, check_errors=False)
    b = torch.linalg.solve_triangular(c, shifted.T, upper=False).T
    bb = b.T @ b
    values, vectors = block_eigh(((bb + bb.T) * 0.5)[None])
    # block_eigh pads odd dimensions; ignore its artificial coordinate.
    vectors = vectors[0, :rank]
    values = values[0]
    positive = values > torch.finfo(values.dtype).tiny
    u = (b @ vectors) / values.clamp_min(torch.finfo(values.dtype).tiny).sqrt()
    u = torch.where(positive, u, 0)
    eigenvalues = (values - nu).clamp_min(0)
    # Ignore a possible padded zero eigenvalue when choosing lambda_min.
    smallest = torch.where(positive, eigenvalues, torch.inf).min()
    smallest = torch.where(torch.isfinite(smallest), smallest, 0)
    weights = (smallest + damping) / (eigenvalues + damping) - 1
    weights = torch.where(positive, weights, 0)
    orth = u.T @ u - torch.diag(positive.to(u.dtype))
    valid = (
        (info == 0)
        & torch.isfinite(u).all()
        & torch.isfinite(weights).all()
        & ((omega.T @ omega - eye).norm() <= 1e-9 * rank)
        & (orth.norm() <= 1e-7 * rank)
    )
    return u, weights, valid


def apply(residual, u, weights):
    x = residual.double().flatten()
    return (x + u @ (weights * (u.T @ x))).reshape_as(residual).to(residual.dtype)
