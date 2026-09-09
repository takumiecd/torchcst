"""Regularized atom-block update without a radius constraint or clipping."""

import torch

from torchcst._derivatives.local_solve import solve_blocks
from torchcst._runtime.validation import require
from torchcst.optim.solvers.quartic import QuarticSolveResult


def solve(metric, linear, lr, damping, *, rtol=1e-5):
    displacement, _relative, valid = solve_blocks(
        metric, -lr * linear, damping, rtol=rtol
    )
    require(valid, "unconstrained block update failed", FloatingPointError)
    matrix = (metric.double() + metric.double().transpose(-1, -2)) * 0.5
    matrix = matrix + damping * torch.eye(
        matrix.shape[-1], device=matrix.device, dtype=matrix.dtype
    )
    hd = (matrix @ displacement.double().unsqueeze(-1)).squeeze(-1) / lr
    objective = (linear.double() * displacement).sum() + 0.5 * (
        displacement.double() * hd
    ).sum()
    residual = (linear.double() + hd).norm()
    return QuarticSolveResult(displacement, objective, residual, 0, 0, 1, valid, False)
