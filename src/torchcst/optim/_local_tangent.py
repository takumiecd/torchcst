"""Explicit diagonal / atom-block approximations of the update quadratic.

The visible second moment and cross-atom moment compression remain unchanged.
All blocks share one Euclidean ball and one secular multiplier.
"""

from dataclasses import dataclass

import torch

from torchcst._runtime.validation import require

from .solvers.quartic import QuarticSolveResult


@dataclass(frozen=True)
class LocalTangentResult(QuarticSolveResult):
    shift: torch.Tensor
    relative_residual: torch.Tensor
    relative_complementarity: torch.Tensor


def solve(problem, *, approximation, radius, rtol):
    if approximation not in ("diagonal", "atom_block"):
        raise ValueError("expected diagonal or atom_block update approximation")
    linear = problem.linear.double()
    matrix = (
        problem.operator.diagonal()
        if approximation == "diagonal"
        else problem.operator.blocks()
    )
    finite = torch.isfinite(matrix).all() & torch.isfinite(linear).all()
    require(finite, "non-finite local tangent objective", FloatingPointError)
    matrix, linear = torch.where(finite, matrix, 0), torch.where(finite, linear, 0)
    if approximation == "diagonal":
        values, coeff = matrix, -linear
        vectors = None
    else:
        from torchcst._derivatives.block_eigh import block_eigh

        values, vectors = block_eigh(matrix)
        padding = vectors.shape[-1] - linear.shape[-1]
        padded = torch.nn.functional.pad(matrix, (0, padding, 0, padding))
        residual = (padded @ vectors - vectors * values[:, None, :]).norm(dim=(-2, -1))
        orthogonal = (
            vectors.transpose(-1, -2) @ vectors
            - torch.eye(vectors.shape[-1], device=matrix.device, dtype=matrix.dtype)
        ).norm(dim=(-2, -1))
        require(
            (residual <= 1e-9 * padded.norm(dim=(-2, -1)).clamp_min(1e-30)).all()
            & (orthogonal <= 1e-9 * vectors.shape[-1]).all(),
            "local tangent eigensolve residual failed",
            FloatingPointError,
        )
        rhs = torch.nn.functional.pad(-linear, (0, padding))
        coeff = (vectors.transpose(-1, -2) @ rhs[..., None]).squeeze(-1)
    require(
        (values >= -1e-12 * matrix.abs().amax().clamp_min(1e-30)).all(),
        "local tangent metric is not positive semidefinite",
        FloatingPointError,
    )
    # Only roundoff-negative eigenvalues/diagonal entries may be clamped.
    values = values.clamp_min(0)
    if matrix.is_cuda:
        from torchcst._derivatives.tangent_triton import spectral_solution

        solution, shift = spectral_solution(values.flatten(), coeff.flatten(), radius)
        solution = solution.reshape_as(coeff)
    else:
        from ._tangent_device import spectral_ball

        solution, shift = spectral_ball(values, coeff, radius)
    d = solution if vectors is None else (vectors @ solution[..., None]).squeeze(-1)
    d = d[..., : linear.shape[-1]].to(problem.linear)
    d = d * (radius / d.norm().clamp_min(1e-30)).clamp(max=1)
    actual = d.double()
    hd = (
        matrix * actual if vectors is None else (matrix @ actual[..., None]).squeeze(-1)
    )
    gradient = linear + hd
    norm_b = linear.norm().clamp_min(1e-300)
    residual = (gradient + shift * actual).norm()
    relative = residual / norm_b
    complementarity = shift * (radius - actual.norm()).abs() / norm_b
    valid = (
        torch.isfinite(actual).all()
        & torch.isfinite(relative)
        & (relative <= rtol)
        & (complementarity <= rtol)
        & (actual.norm() <= radius * (1 + 10 * torch.finfo(d.dtype).eps))
    )
    require(valid, "local tangent solve did not converge", FloatingPointError)
    return LocalTangentResult(
        d.detach(),
        ((linear * actual).sum() + 0.5 * (actual * hd).sum()).detach(),
        residual.detach(),
        0,
        80,
        1,
        valid.detach(),
        (actual.norm() >= radius * (1 - 1e-6)).detach(),
        shift.detach(),
        relative.detach(),
        complementarity.detach(),
    )
