"""Exact ray objective contractions through factor-local derivatives."""

from functools import cache

import torch

from torchcst._derivatives import factored_taylor as ft
from torchcst._runtime.graphs import CapturedCall

from ._ray import unit_quartic_minimum
from .device import _project


def value_gradient(a, b, factors, metric, inverse_lr, x):
    represented = ft.displacement(factors, x)
    force = metric * represented
    value = (a * x).sum() + 0.5 * torch.einsum("kp,kpq,kq->", x, b, x)
    value = value + 0.5 * inverse_lr * (represented * force).sum()
    gradient = (
        a
        + torch.einsum("kpq,kq->kp", b, x)
        + inverse_lr * ft.pullback(factors, force, x)
    )
    return value, gradient


def coefficients(a, b, f, metric, inverse_lr, x, v):
    base = ft.displacement(f, x)
    first = ft.tangent(f, v) + ft.second(f, x, v)
    second = 0.5 * ft.second(f, v, v)
    return torch.stack(
        (
            (a * v).sum()
            + torch.einsum("kp,kpq,kq->", x, b, v)
            + inverse_lr * (metric * base * first).sum(),
            0.5 * torch.einsum("kp,kpq,kq->", v, b, v)
            + 0.5 * inverse_lr * (metric * (first.square() + 2 * base * second)).sum(),
            inverse_lr * (metric * first * second).sum(),
            0.5 * inverse_lr * (metric * second.square()).sum(),
        )
    )


def run(args, radius, corrections, tolerance):
    a, b = args[:2]
    f = args[2:8]
    row, column, inverse_lr, eps = args[8:]
    metric = row[:, None] * column[None, :] + eps
    diagonal = b.diagonal(dim1=-2, dim2=-1).abs() + inverse_lr * ft.metric_diagonal(
        f, row, column, eps
    )
    diagonal = diagonal.clamp_min(diagonal.amax() * 1e-6 + 1e-12)
    x, value, gradient = torch.zeros_like(a), a.new_zeros(()), a
    count = torch.zeros((), dtype=torch.int64, device=a.device)
    for _ in range(corrections):
        direction = -gradient / diagonal
        direction = direction * (radius / direction.norm().clamp_min(1e-30))
        chord = _project(x + direction, radius) - x
        fallback = (
            _project(x - radius * gradient / gradient.norm().clamp_min(1e-30), radius)
            - x
        )
        chord = torch.where((chord * gradient).sum() < 0, chord, fallback)
        scalar = unit_quartic_minimum(
            coefficients(a, b, f, metric, inverse_lr, x, chord)
        ).to(x.dtype)
        candidate = _project(x + scalar * chord, radius)
        candidate_value, candidate_gradient = value_gradient(
            a, b, f, metric, inverse_lr, candidate
        )
        accepted = (
            torch.isfinite(candidate_value)
            & torch.isfinite(candidate_gradient).all()
            & (candidate_value < value)
        )
        x = torch.where(accepted, candidate, x)
        value = torch.where(accepted, candidate_value, value)
        gradient = torch.where(accepted, candidate_gradient, gradient)
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
def runner(radius, corrections, tolerance):
    function = torch.compile(
        lambda *args: run(args, radius, corrections, tolerance),
        fullgraph=True,
        dynamic=False,
    )
    return CapturedCall(function)


def solve(problem, radius, corrections, tolerance):
    from ..moments.second import SeparableDiagonalMetric

    if not isinstance(problem.moments.second.metric, SeparableDiagonalMetric):
        raise TypeError("factored ray requires a separable metric")
    point = problem.context.current_point
    factors = problem.context.geometry.factor_local_derivatives(point)
    row, column, eps = problem.moments.second.metric.separable_weights()
    args = (
        problem._first_constant,
        problem._first_linear,
        *factors,
        row,
        column,
        point.new_full((), 1 / problem.learning_rate),
        point.new_full((), eps),
    )
    if point.device.type == "cuda":
        return runner(radius, corrections, tolerance)(*args)
    return run(args, radius, corrections, tolerance)
