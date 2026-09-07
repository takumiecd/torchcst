"""Fixed-budget projected BFGS with device-resident adaptive state."""

import math
from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall
from torchcst._runtime.validation import require

from ..problem import QuarticProblem
from ._compiled import CompiledQuarticModel, visible_value_gradient
from .base import QuarticSolver
from .quartic import QuarticSolveResult


def _project(x, radius):
    return x * (radius / x.norm().clamp_min(torch.finfo(x.dtype).tiny)).clamp(max=1)


def _trial(
    a,
    b,
    j,
    h,
    metric,
    inverse_lr,
    x,
    value,
    grad,
    inverse,
    direction,
    length,
    accepted_count,
    retries,
    radius,
    max_iter,
    tolerance,
):
    candidate = _project(x + length * direction, radius)
    new_value, new_grad = visible_value_gradient(
        a, b, j, h, metric, inverse_lr, candidate
    )
    step = candidate - x
    residual = (x - _project(x - grad, radius)).norm()
    active = (accepted_count < max_iter) & (residual > tolerance)
    accepted = active & torch.isfinite(new_value) & torch.isfinite(new_grad).all()
    accepted = (
        accepted
        & (new_value < value)
        & (new_value <= value + 1e-4 * (grad * step).sum())
    )
    s, y = step.flatten(), (new_grad - grad).flatten()
    sy = s @ y
    use_bfgs = accepted & (sy > 1e-6 * s.norm() * y.norm())
    safe_sy = torch.where(use_bfgs, sy, torch.ones_like(sy))
    hy = inverse @ y
    update = inverse + ((safe_sy + y @ hy) / safe_sy.square()) * torch.outer(s, s)
    update = update - (torch.outer(hy, s) + torch.outer(s, hy)) / safe_sy
    inverse = torch.where(use_bfgs, update, inverse)
    x = torch.where(accepted, candidate, x)
    value = torch.where(accepted, new_value, value)
    grad = torch.where(accepted, new_grad, grad)
    count = accepted_count + accepted.to(accepted_count.dtype)
    retries = torch.where(accepted, 0, retries + active.to(retries.dtype))
    reset = retries >= 12
    scale = radius / grad.norm().clamp_min(torch.finfo(grad.dtype).tiny)
    inverse = torch.where(
        reset, torch.eye(x.numel(), device=x.device, dtype=x.dtype) * scale, inverse
    )
    proposed = _project(x - (inverse @ grad.flatten()).reshape_as(x), radius) - x
    fallback = _project(x - scale * grad, radius) - x
    proposed = torch.where((grad * proposed).sum() < 0, proposed, fallback)
    direction = torch.where(accepted | reset, proposed, direction)
    length = torch.where(accepted | reset, 1, length * 0.5)
    retries = torch.where(reset, 0, retries)
    return x, value, grad, inverse, direction, length, count, retries


@cache
def _compiled_trial():
    return torch.compile(_trial, fullgraph=True, dynamic=False)


def _run(coefficients, radius, max_iter, evaluations, tolerance, *, compiled):
    x = torch.zeros_like(coefficients[0])
    value, grad = visible_value_gradient(*coefficients, x)
    scale = radius / grad.norm().clamp_min(torch.finfo(grad.dtype).tiny)
    inverse = torch.eye(x.numel(), device=x.device, dtype=x.dtype) * scale
    direction = _project(-scale * grad, radius)
    one = x.new_ones(())
    zero = torch.zeros((), dtype=torch.int64, device=x.device)
    state = (x, value, grad, inverse, direction, one, zero, zero.clone())
    trial = _compiled_trial() if compiled else _trial
    for _ in range(evaluations - 1):
        state = trial(*coefficients, *state, radius, max_iter, tolerance)
    x, value, grad, _, _, _, count, _ = state
    residual = (x - _project(x - grad, radius)).norm()
    return (
        x,
        value,
        residual,
        count,
        residual <= tolerance,
        x.norm() >= radius * (1 - 1e-4),
    )


@cache
def _runner(radius, max_iter, evaluations, tolerance):
    return CapturedCall(
        lambda *args: _run(
            args, radius, max_iter, evaluations, tolerance, compiled=True
        )
    )


class DeviceBFGS(QuarticSolver):
    """One zero start, exact quartic, projected BFGS; diagnostics stay on device.

    The fixed evaluation schedule uses masked state updates. This is a different
    search algorithm from BallNewton; it provides first-order stationarity only.
    """

    supports_deferred_execution = True

    def __init__(self, *, max_iter=30, max_evaluations=150, tolerance_grad=1e-5):
        for value in (max_iter, max_evaluations):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "iteration/evaluation budgets must be positive integers"
                )
        if not math.isfinite(tolerance_grad) or tolerance_grad <= 0:
            raise ValueError("tolerance_grad must be finite and positive")
        self.max_iter = max_iter
        self.max_evaluations = max_evaluations
        self.tolerance_grad = tolerance_grad

    def solve(self, problem, *, trust_radius):
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if not math.isfinite(trust_radius) or trust_radius <= 0:
            raise ValueError("trust_radius must be finite and positive")
        coefficients = CompiledQuarticModel(problem).coefficients
        if coefficients[0].dtype not in (torch.float32, torch.float64):
            raise ValueError("DeviceBFGS requires float32/float64")
        with torch.no_grad():
            args = (
                trust_radius,
                self.max_iter,
                self.max_evaluations,
                self.tolerance_grad,
            )
            if coefficients[0].device.type == "cuda":
                result = _runner(*args)(*coefficients)
            else:
                result = _run(coefficients, *args, compiled=False)
            x, value, residual, count, converged, boundary = result
            require(
                torch.isfinite(value) & torch.isfinite(x).all(),
                "device BFGS returned non-finite state",
                FloatingPointError,
            )
            return QuarticSolveResult(
                x, value, residual, 0, count, self.max_evaluations, converged, boundary
            )
