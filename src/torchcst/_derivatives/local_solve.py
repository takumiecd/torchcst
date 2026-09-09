"""Batched small positive-definite solves with device-side diagnostics."""

import torch


def solve_blocks(gram, rhs, damping, *, rtol=1e-5):
    matrix = (gram.double() + gram.double().transpose(-1, -2)) * 0.5
    matrix = matrix + damping * torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    )
    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    solution = (
        torch.cholesky_solve(rhs.double().unsqueeze(-1), factor).squeeze(-1).to(rhs)
    )
    residual = (matrix @ solution.double().unsqueeze(-1)).squeeze(-1) - rhs.double()
    relative = residual.norm(dim=-1) / rhs.double().norm(dim=-1).clamp_min(1e-300)
    valid = (
        (info == 0).all() & torch.isfinite(solution).all() & (relative <= rtol).all()
    )
    return solution, relative.amax(), valid
