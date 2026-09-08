"""Eager block-preconditioned CG for explicitly damped tangent compression."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressionResult:
    iterations: int
    relative_residual: float
    converged: bool
    backend: str


class CompressionError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(
            f"tangent recompression failed after {result.iterations} iterations: "
            f"true relative residual {result.relative_residual:.3g}"
        )


def validate_pcg(*, damping, max_iter, rtol):
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("PCG requires positive finite first_moment_damping")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("recompression_max_iter must be a positive integer")
    if not math.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("recompression_rtol must lie between zero and one")


@torch.no_grad()
def solve_compression(prepared, rhs, *, damping, max_iter=64, rtol=1e-5):
    """Solve (J.T J + damping I) alpha = rhs without a global Gram matrix.

    Uses parameter dtype for vectors and operator actions, FP64 scalar reductions.
    Tests the true residual of the actual returned vector, not a recurrence.
    Native dtype limits attainable residuals; failure never silently adds damping.
    This eager implementation synchronizes for convergence decisions on CUDA.
    """
    validate_pcg(damping=damping, max_iter=max_iter, rtol=rtol)
    prepared._vector(rhs)
    if not torch.isfinite(rhs).all():
        raise ValueError("recompression right-hand side must be finite")

    def dot(x, y):
        return (x.double() * y.double()).sum()

    norm = dot(rhs, rhs).sqrt()
    alpha = torch.zeros_like(rhs)
    if norm == 0:
        return alpha, CompressionResult(0, 0.0, True, prepared.backend)

    blocks = prepared.gram_blocks()
    blocks = (blocks + blocks.transpose(-1, -2)) * 0.5
    blocks.diagonal(dim1=-1, dim2=-2).add_(damping)
    factor, info = torch.linalg.cholesky_ex(blocks)
    if (info != 0).any() or not torch.isfinite(factor).all():
        raise CompressionError(CompressionResult(0, math.inf, False, prepared.backend))

    def precondition(r):
        return torch.cholesky_solve(r[..., None], factor).squeeze(-1)

    def action(x):
        return prepared.gram_matvec(x) + damping * x

    residual = rhs.clone()
    z = precondition(residual)
    direction = z.clone()
    rz = dot(residual, z)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        ad = action(direction)
        curvature = dot(direction, ad)
        if not torch.isfinite(curvature) or curvature <= 0:
            break
        step = (rz / curvature).to(rhs.dtype)
        alpha.add_(step * direction)
        residual.sub_(step * ad)
        # Replace drifted recurrence with the actual residual before accepting.
        if dot(residual, residual).sqrt() <= rtol * norm:
            residual = rhs - action(alpha)
            relative = float(dot(residual, residual).sqrt() / norm)
            if math.isfinite(relative) and relative <= rtol:
                return alpha, CompressionResult(
                    iteration, relative, True, prepared.backend
                )
            z = precondition(residual)
            direction = z.clone()
            rz = dot(residual, z)
            continue
        z = precondition(residual)
        rz_next = dot(residual, z)
        direction = z + (rz_next / rz).to(rhs.dtype) * direction
        rz = rz_next
    residual = rhs - action(alpha)
    relative = float(dot(residual, residual).sqrt() / norm)
    result = CompressionResult(
        iteration,
        relative,
        math.isfinite(relative) and relative <= rtol,
        prepared.backend,
    )
    if not result.converged:
        raise CompressionError(result)
    return alpha, result
