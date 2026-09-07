"""Few feasible directions, with scalar quartic minimization on each chord."""

import math
from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall
from torchcst._runtime.validation import require

from ..problem import QuarticProblem
from ._compiled import CompiledQuarticModel, visible_value_gradient
from ._ray import ray_coefficients, unit_quartic_minimum
from .base import QuarticSolver
from .device import _project
from .quartic import QuarticSolveResult


def _ray_step(a, b, j, h, metric, inverse_lr, x, value, gradient, diagonal, radius):
    direction = -gradient / diagonal
    direction = direction * (radius / direction.norm().clamp_min(1e-30))
    chord = _project(x + direction, radius) - x
    fallback = (
        _project(x - radius * gradient / gradient.norm().clamp_min(1e-30), radius) - x
    )
    chord = torch.where((chord * gradient).sum() < 0, chord, fallback)
    coefficients = ray_coefficients(a, b, j, h, metric, inverse_lr, x, chord)
    t = unit_quartic_minimum(coefficients).to(x.dtype)
    candidate = _project(x + t * chord, radius)
    candidate_value, candidate_gradient = visible_value_gradient(
        a, b, j, h, metric, inverse_lr, candidate
    )
    accepted = (
        torch.isfinite(candidate_value)
        & torch.isfinite(candidate_gradient).all()
        & (candidate_value < value)
    )
    return (
        torch.where(accepted, candidate, x),
        torch.where(accepted, candidate_value, value),
        torch.where(accepted, candidate_gradient, gradient),
        accepted,
    )


@cache
def _compiled_step():
    return torch.compile(_ray_step, fullgraph=True, dynamic=False)


def _run(coefficients, radius, corrections, tolerance, compiled):
    a, b, j, _h, metric, inverse_lr = coefficients
    x = torch.zeros_like(a)
    value, gradient = visible_value_gradient(*coefficients, x)
    diagonal = b.diagonal(dim1=-2, dim2=-1).abs() + inverse_lr * torch.einsum(
        "kmp,m,kmp->kp", j, metric, j
    )
    diagonal = diagonal.clamp_min(diagonal.amax() * 1e-6 + 1e-12)
    count = torch.zeros((), device=x.device, dtype=torch.int64)
    step = _compiled_step() if compiled else _ray_step
    for _ in range(corrections):
        x, value, gradient, accepted = step(
            *coefficients, x, value, gradient, diagonal, radius
        )
        count = count + accepted
    residual = (x - _project(x - gradient, radius)).norm()
    return (
        x,
        value,
        residual,
        count,
        residual <= tolerance,
        x.norm() >= radius * (1 - 1e-4),
    )


@cache
def _runner(radius, corrections, tolerance):
    return CapturedCall(lambda *args: _run(args, radius, corrections, tolerance, True))


class DeviceRay(QuarticSolver):
    """One zero start and a fixed number of exact scalar-quartic corrections.

    This does not solve the full multidimensional quartic to convergence.
    Original-objective checks retain the best feasible finite point.
    """

    supports_deferred_execution = True

    def __init__(self, *, corrections=4, tolerance_grad=1e-5):
        if (
            isinstance(corrections, bool)
            or not isinstance(corrections, int)
            or corrections < 1
        ):
            raise ValueError("corrections must be a positive integer")
        if not math.isfinite(tolerance_grad) or tolerance_grad <= 0:
            raise ValueError("tolerance_grad must be finite and positive")
        self.corrections = corrections
        self.tolerance_grad = tolerance_grad

    def solve(self, problem, *, trust_radius):
        if not isinstance(problem, QuarticProblem):
            raise TypeError("problem must be a QuarticProblem")
        if not math.isfinite(trust_radius) or trust_radius <= 0:
            raise ValueError("invalid trust radius")
        coefficients = CompiledQuarticModel(problem).coefficients
        if coefficients[0].dtype not in (torch.float32, torch.float64):
            raise ValueError("DeviceRay requires float32/float64")
        with torch.no_grad():
            if coefficients[0].device.type == "cuda":
                out = _runner(trust_radius, self.corrections, self.tolerance_grad)(
                    *coefficients
                )
            else:
                out = _run(
                    coefficients,
                    trust_radius,
                    self.corrections,
                    self.tolerance_grad,
                    False,
                )
            x, value, residual, count, converged, boundary = out
            require(
                torch.isfinite(x).all() & torch.isfinite(value),
                "non-finite ray solver result",
                FloatingPointError,
            )
            return QuarticSolveResult(
                x, value, residual, 0, count, self.corrections + 1, converged, boundary
            )
