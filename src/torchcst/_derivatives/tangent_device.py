"""Tensor-controlled PCG: no iteration-dependent host reads or library solves."""

from functools import cache

import torch

from torchcst._runtime.graphs import CapturedCall


def block_inverse(blocks, damping):
    """Small SPD blocks, FP64 Cholesky and inverse; all checks stay on device."""
    a = (blocks.double() + blocks.double().transpose(-1, -2)) * 0.5
    q = a.shape[-1]
    a = a + damping * torch.eye(q, device=a.device, dtype=a.dtype)
    factor = torch.zeros_like(a)
    valid = torch.isfinite(a).all()
    for j in range(q):
        pivot = a[:, j, j] - factor[:, j, :j].square().sum(-1)
        valid = valid & (pivot > 0).all() & torch.isfinite(pivot).all()
        factor[:, j, j] = torch.where(pivot > 0, pivot, 1).sqrt()
        for i in range(j + 1, q):
            factor[:, i, j] = (
                a[:, i, j] - (factor[:, i, :j] * factor[:, j, :j]).sum(-1)
            ) / factor[:, j, j]
    inverse = torch.zeros_like(a)
    for i in range(q):
        rhs = torch.zeros_like(a[:, i])
        rhs[:, i] = 1
        inverse[:, i] = (
            rhs - (factor[:, i, :i, None] * inverse[:, :i]).sum(1)
        ) / factor[:, i, i, None]
    result = inverse.transpose(-1, -2) @ inverse
    return result.to(blocks.dtype), valid & torch.isfinite(result).all()


def dot(a, b):
    return (a.double() * b.double()).sum()


def advance(alpha, residual, direction, ad, rz, active, healthy, count, norm, rtol):
    curvature = dot(direction, ad)
    good = torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(rz)
    healthy = healthy & (~active | good)
    working = active & healthy
    step = torch.where(working, rz / torch.where(good, curvature, 1), 0).to(alpha.dtype)
    alpha = alpha + step * direction
    residual = residual - step * ad
    candidate = working & (dot(residual, residual).sqrt() <= rtol * norm)
    return alpha, residual, working, healthy, count + active.to(count.dtype), candidate


def precondition(residual, inverse, lowrank=None, weights=None):
    if lowrank is None:
        return (inverse @ residual[..., None]).squeeze(-1)
    from .tangent_nystrom import apply

    return apply(residual, lowrank, weights)


def finish_iteration(
    rhs,
    alpha,
    residual,
    direction,
    aa,
    inverse,
    rz,
    active,
    healthy,
    candidate,
    norm,
    rtol,
    lowrank=None,
    weights=None,
):
    exact = rhs - aa
    residual = torch.where(candidate, exact, residual)
    converged = candidate & (dot(exact, exact).sqrt() <= rtol * norm)
    z = precondition(residual, inverse, lowrank, weights)
    rz_next = dot(residual, z)
    active = active & healthy & ~converged
    beta = torch.where(
        active & ~candidate, rz_next / torch.where(rz != 0, rz, 1), 0
    ).to(rhs.dtype)
    direction = torch.where(active, z + beta * direction, 0)
    return residual, direction, rz_next, active


@cache
def _compiled(function):
    return torch.compile(function, fullgraph=True, dynamic=False)


def pcg(prepared, rhs, *, damping, max_iter, rtol, compiled=False, nystrom=None):
    inverse_fn = _compiled(block_inverse) if compiled else block_inverse
    step_fn = _compiled(advance) if compiled else advance
    finish_fn = _compiled(finish_iteration) if compiled else finish_iteration
    lowrank, weights = None, None
    if nystrom is None:
        inverse, healthy = inverse_fn(prepared.gram_blocks(), damping)
    else:
        lowrank, weights, healthy = nystrom
        inverse = None
    healthy = healthy & torch.isfinite(rhs).all()
    norm = dot(rhs, rhs).sqrt()
    active = healthy & (norm > 0)
    alpha = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = precondition(residual, inverse, lowrank, weights)
    direction = torch.where(active, direction, 0)
    rz = dot(residual, direction)
    count = torch.zeros((), device=rhs.device, dtype=torch.int32)
    for _ in range(max_iter):
        ad = prepared._device_gram(direction, damping=damping, active=active)
        alpha, residual, active, healthy, count, candidate = step_fn(
            alpha, residual, direction, ad, rz, active, healthy, count, norm, rtol
        )
        aa = prepared._device_gram(alpha, damping=damping, active=candidate)
        residual, direction, rz, active = finish_fn(
            rhs,
            alpha,
            residual,
            direction,
            aa,
            inverse,
            rz,
            active,
            healthy,
            candidate,
            norm,
            rtol,
            lowrank,
            weights,
        )
    true_residual = rhs - prepared._device_gram(alpha, damping=damping)
    relative = dot(true_residual, true_residual).sqrt() / norm.clamp_min(1e-300)
    valid = (
        healthy
        & torch.isfinite(alpha).all()
        & torch.isfinite(relative)
        & (relative <= rtol)
    )
    return alpha, count, relative, valid


@cache
def _runner(damping, max_iter, rtol, gram_action):
    # Bind no changing parameter/factor values: every replay receives fresh tensors.
    def run(point, u, v, du, dv, rhs):
        from types import SimpleNamespace

        from .tangent_ops import PreparedFactors

        prepared = PreparedFactors(
            SimpleNamespace(
                backend="specialized", execution="triton", gram_action=gram_action
            ),
            point,
            (u, v, du, dv),
        )
        return pcg(
            prepared, rhs, damping=damping, max_iter=max_iter, rtol=rtol, compiled=True
        )

    return CapturedCall(run)


def solve(prepared, rhs, *, damping, max_iter, rtol):
    if rhs.is_cuda:
        return _runner(
            damping, max_iter, rtol, getattr(prepared._ops, "gram_action", "pair")
        )(prepared._point, prepared._u, prepared._v, prepared._du, prepared._dv, rhs)
    return pcg(prepared, rhs, damping=damping, max_iter=max_iter, rtol=rtol)
