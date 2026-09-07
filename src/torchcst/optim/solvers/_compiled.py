"""Tensor-only Newton kernels. Step-dependent coefficients are explicit inputs."""

from __future__ import annotations

from functools import cache

import torch

from ..moments.second import SeparableDiagonalMetric


def visible_value_gradient(a, b, j, h, metric, inverse_lr, d):
    represented = torch.einsum("kmp,kp->m", j, d) + 0.5 * torch.einsum(
        "kmpq,kp,kq->m", h, d, d
    )
    force = metric * represented
    value = (a * d).sum() + 0.5 * torch.einsum("kp,kpq,kq->", d, b, d)
    value = value + 0.5 * inverse_lr * (represented * force).sum()
    frame = j + torch.einsum("kmpq,kq->kmp", h, d)
    gradient = a + torch.einsum("kpq,kq->kp", b, d)
    gradient = gradient + inverse_lr * torch.einsum("kmp,m->kp", frame, force)
    return value, gradient


def visible_hessian(a, b, j, h, metric, inverse_lr, d):
    atoms, parameters = d.shape
    size = d.numel()
    frame = j + torch.einsum("kmpq,kq->kmp", h, d)
    columns = frame.permute(1, 0, 2).reshape(-1, size)
    matrix = (columns.T @ (columns.T * metric).T) * inverse_lr
    represented = torch.einsum("kmp,kp->m", j, d) + 0.5 * torch.einsum(
        "kmpq,kp,kq->m", h, d, d
    )
    blocks = b + inverse_lr * torch.einsum("m,kmpq->kpq", metric * represented, h)
    index = torch.arange(atoms, device=d.device)
    result = matrix.reshape(atoms, parameters, atoms, parameters)
    result[index, :, index, :] += blocks
    return 0.5 * (matrix + matrix.T)


def spectral_ball_device(eigenvalues, eigenvectors, rhs, radius):
    """Same spectral subproblem, including PSD nullspaces and the hard case.

    Scalar bisection remains in device float64, matching the host-double
    reference arithmetic. There are no Tensor -> Python conversions. Fixed
    iterations with a tensor convergence mask preserve the reference bracket.
    """
    values = eigenvalues.double()
    coefficients = (eigenvectors.T @ rhs).double()
    scale = values.abs().amax().clamp_min(1e-30)
    eps = torch.finfo(eigenvalues.dtype).eps
    lower = (-values[0]).clamp_min(0)
    shifted = values + lower
    null = shifted <= 8 * eps * scale
    endpoint = torch.where(null, 0.0, coefficients / shifted.clamp_min(1e-300))
    endpoint_square = endpoint.square().sum()
    rhs_norm = coefficients.square().sum().sqrt()
    small_null_rhs = (
        (~null) | (coefficients.abs() <= 8 * eps * rhs_norm.clamp_min(1e-30))
    ).all()
    endpoint_valid = small_null_rhs & (endpoint_square <= radius * radius)
    addition = torch.copysign(
        (radius * radius - endpoint_square).clamp_min(0).sqrt(), coefficients[0]
    )
    first = torch.where(lower > 0, addition, endpoint[0])
    endpoint = torch.cat((first.reshape(1), endpoint[1:]))
    upper = lower + rhs_norm / radius
    active = torch.ones((), device=values.device, dtype=torch.bool)
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        norm = (coefficients / (values + midpoint).clamp_min(1e-300)).norm()
        go_right = norm > radius
        lower = torch.where(active & go_right, midpoint, lower)
        upper = torch.where(active & ~go_right, midpoint, upper)
        active = active & ((upper - lower) > 2e-14 * torch.maximum(scale, upper.abs()))
    solution = coefficients / (values + upper).clamp_min(1e-300)
    result = eigenvectors @ solution.to(eigenvalues.dtype)
    result = result * torch.clamp(
        radius / result.norm().clamp_min(torch.finfo(result.dtype).tiny), max=1.0
    )
    endpoint_result = eigenvectors @ endpoint.to(eigenvalues.dtype)
    return torch.where(endpoint_valid, endpoint_result, result)


@cache
def compiled_kernel(name, backend="inductor", mode="reduce-overhead"):
    # One function per kernel/runtime configuration, reused across optimizer steps.
    function = globals()[name]
    return torch.compile(
        function, fullgraph=True, dynamic=False, backend=backend, mode=mode
    )


def call_compiled(name, *args, backend="inductor", mode="reduce-overhead"):
    if args[0].device.type == "cuda":
        torch.compiler.cudagraph_mark_step_begin()
    result = compiled_kernel(name, backend, mode)(*args)
    # Retained solver state must not alias CUDA Graph replay output buffers.
    if isinstance(result, tuple):
        return tuple(value.clone() for value in result)
    return result.clone()


class CompiledQuarticModel:
    def __init__(self, problem, *, backend="inductor", mode="reduce-overhead"):
        if not isinstance(problem.moments.second.metric, SeparableDiagonalMetric):
            raise TypeError("compiled Newton requires a separable diagonal metric")
        local = problem.context.geometry.local_quadratic_derivatives(
            problem.context.current_point
        )
        if local is None:
            raise ValueError("compiled Newton requires materialized local derivatives")
        self.coefficients = (
            problem._first_constant.detach(),
            problem._first_linear.detach(),
            *(value.detach() for value in local),
            problem.moments.second.metric.diagonal().reshape(-1).detach(),
            problem.context.current_point.new_tensor(1.0 / problem.learning_rate),
        )
        self.backend, self.mode = backend, mode

    def value_and_gradient(self, d):
        return call_compiled(
            "visible_value_gradient",
            *self.coefficients,
            d,
            backend=self.backend,
            mode=self.mode,
        )

    def hessian(self, d):
        return call_compiled(
            "visible_hessian",
            *self.coefficients,
            d,
            backend=self.backend,
            mode=self.mode,
        )


def newton_status(eigenvalues, current, gradient, threshold, radius):
    scale = eigenvalues.abs().amax().double().clamp_min(1e-12)
    projected = current - gradient
    projected = projected * torch.clamp(
        radius / projected.norm().clamp_min(torch.finfo(current.dtype).tiny), max=1.0
    )
    residual = (current - projected).norm()
    interior = current.norm().double() < radius * (1 - 1e-5)
    negative = eigenvalues[0].double() < -8 * torch.finfo(current.dtype).eps * scale
    return scale, (residual.double() <= threshold) & ~(interior & negative)


def accept_candidate(
    gradient,
    step,
    matrix,
    regularization,
    value,
    candidate_value,
    candidate_gradient,
    ratio,
):
    step = step.reshape(-1)
    regularization = regularization.to(step.dtype)
    predicted = (
        -(gradient.reshape(-1) @ step)
        - 0.5 * (step @ (matrix @ step))
        - 0.5 * regularization * (step @ step)
    )
    actual = value - candidate_value
    return (
        torch.isfinite(candidate_value)
        & torch.isfinite(candidate_gradient).all()
        & (predicted > 0)
        & (actual > 0)
        & (actual >= ratio * predicted)
    )
