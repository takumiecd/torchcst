"""Device-only solve of explicitly damped positive-definite Gram systems."""

from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall


def solve_damped(matrix, rhs):
    # The caller has already applied its explicit damping. Retain GPU FP64
    # accumulation, but never hide a rank cutoff or choose damping here.
    a, b = matrix.double(), rhs.double()
    factor, info = torch.linalg.cholesky_ex(a, check_errors=False)
    y = torch.linalg.solve_triangular(factor, b[:, None], upper=False)
    x = torch.linalg.solve_triangular(factor.T, y, upper=True).squeeze(1)
    residual = (a @ x - b).norm()
    scale = a.norm() * x.norm() + b.norm()
    valid = (
        (info == 0)
        & torch.isfinite(x).all()
        & (
            residual
            <= 64 * a.shape[0] * torch.finfo(a.dtype).eps * scale.clamp_min(1e-30)
        )
    )
    result = x.to(matrix.dtype)
    return result, valid & torch.isfinite(result).all()


@cache
def _runner():
    return CapturedCall(solve_damped)


def device_solve(matrix, rhs):
    if matrix.device.type == "cuda":
        return _runner()(matrix, rhs)
    return solve_damped(matrix, rhs)
