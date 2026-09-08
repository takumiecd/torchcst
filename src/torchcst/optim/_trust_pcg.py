"""Shifted PCG ball solver with a GPU-resident bracket and KKT certificate.

The zero-shift solve is unpreconditioned CG from zero, preserving the minimum
Euclidean norm in a singular consistent system. Positive shifts use atom-block PCG.
Only vectors and scalar state are retained; no Krylov basis or dense Gram.
"""

from dataclasses import dataclass
from functools import cache

import torch

from torchcst._derivatives.tangent_device import _compiled, block_inverse, dot
from torchcst._runtime.graphs import CapturedCall
from torchcst._runtime.validation import require

from .solvers.quartic import QuarticSolveResult


@dataclass(frozen=True)
class TangentPCGResult(QuarticSolveResult):
    evaluations: torch.Tensor
    shift: torch.Tensor
    relative_residual: torch.Tensor
    relative_complementarity: torch.Tensor
    shift_iterations: torch.Tensor


def advance(x, r, p, ap, rz, active, healthy, count, norm, tolerance, radius, interior):
    curvature = dot(p, ap)
    good = torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(rz)
    healthy = healthy & (~active | good)
    working = active & healthy
    step = torch.where(working, rz / torch.where(good, curvature, 1), 0)
    x = x + step * p
    r = r - step * ap
    hit = (dot(x, x).sqrt() > radius) & interior
    active = working & ~hit
    candidate = active & (dot(r, r).sqrt() <= tolerance * norm)
    return x, r, active, healthy, count + working.to(count.dtype), candidate


def finish(rhs, x, r, p, ax, inverse, rz, active, candidate, norm, tolerance):
    exact = rhs - ax
    r = torch.where(candidate, exact, r)
    active = active & ~(candidate & (dot(exact, exact).sqrt() <= tolerance * norm))
    z = (inverse @ r[..., None]).squeeze(-1)
    rz_next = dot(r, z)
    beta = torch.where(active & ~candidate, rz_next / torch.where(rz != 0, rz, 1), 0)
    p = torch.where(active, z + beta * p, 0)
    return r, p, rz_next, active


def linear_solve(
    action,
    rhs,
    blocks,
    shift,
    initial,
    enabled,
    old_r,
    old_p,
    old_rz,
    reuse,
    *,
    radius,
    max_iter,
    tolerance,
    interior=False,
    compiled=False,
):
    step_fn = _compiled(advance) if compiled else advance
    finish_fn = _compiled(finish) if compiled else finish
    x = initial.clone()
    norm = dot(rhs, rhs).sqrt()
    if interior:
        inverse = torch.eye(rhs.shape[-1], device=rhs.device, dtype=rhs.dtype).expand(
            rhs.shape[0], -1, -1
        )
        inverse_valid = torch.ones((), device=rhs.device, dtype=torch.bool)
    else:
        inverse_fn = _compiled(block_inverse) if compiled else block_inverse
        inverse, inverse_valid = inverse_fn(blocks, shift)
    r = torch.where(reuse, old_r, rhs - (action(x, enabled & ~reuse) + shift * x))
    healthy = inverse_valid & torch.isfinite(rhs).all() & torch.isfinite(inverse).all()
    active = enabled & healthy & (dot(r, r).sqrt() > tolerance * norm)
    z = (inverse @ r[..., None]).squeeze(-1)
    p = torch.where(active, torch.where(reuse, old_p, z), 0)
    rz = torch.where(reuse, old_rz, dot(r, z))
    count = torch.zeros((), device=rhs.device, dtype=torch.int32)
    calls = 2
    for _ in range(max_iter):
        if not rhs.is_cuda and not bool(active):
            break
        ap = action(p, active) + shift * p
        x, r, active, healthy, count, candidate = step_fn(
            x, r, p, ap, rz, active, healthy, count, norm, tolerance, radius, interior
        )
        ax = action(x, candidate) + shift * x
        calls += 2
        r, p, rz, active = finish_fn(
            rhs, x, r, p, ax, inverse, rz, active, candidate, norm, tolerance
        )
    residual = action(x, enabled) + shift * x - rhs
    ok = (
        healthy
        & torch.isfinite(x).all()
        & (dot(residual, residual).sqrt() <= tolerance * norm)
    )
    return x, residual, count, ok, torch.full_like(count, calls), r, p, rz


@cache
def runner(rate, eps, radius, max_iter, tolerance, interior):
    def run(
        point,
        u,
        v,
        du,
        dv,
        row,
        column,
        rhs,
        blocks,
        shift,
        initial,
        enabled,
        old_r,
        old_p,
        old_rz,
        reuse,
        requested_tolerance,
    ):
        from torchcst._derivatives.tangent_metric import FactorMetricAction

        action = FactorMetricAction(
            point, u, v, du, dv, row, column, eps, rate, triton=True
        )
        return linear_solve(
            action,
            rhs,
            blocks,
            shift,
            initial,
            enabled,
            old_r,
            old_p,
            old_rz,
            reuse,
            radius=radius,
            max_iter=max_iter,
            tolerance=requested_tolerance,
            interior=interior,
            compiled=True,
        )

    return CapturedCall(run)


def certificate(x, residual, rhs, shift, radius, tolerance):
    # Certify the actual returned (radius-clipped) vector using linearity. The
    # residual supplied here was recomputed by the full operator, not a CG recurrence.
    norm = dot(x, x).sqrt()
    scale = (radius / norm.clamp_min(1e-300)).clamp(max=1)
    d = scale * x
    residual = scale * residual + (scale - 1) * rhs
    denominator = dot(rhs, rhs).sqrt().clamp_min(1e-300)
    relative = dot(residual, residual).sqrt() / denominator
    comp = shift * (radius - dot(d, d).sqrt()).abs() / denominator
    valid = (
        torch.isfinite(d).all()
        & torch.isfinite(relative)
        & torch.isfinite(comp)
        & (relative <= tolerance)
        & (comp <= tolerance)
        & (shift >= 0)
    )
    return d, relative, comp, valid


@torch.no_grad()
def solve(problem, *, radius, max_iter=512, shift_steps=32, rtol=1e-5):
    action = problem.operator
    rhs = -problem.linear.double().contiguous()
    blocks = action.blocks()
    require(
        torch.isfinite(rhs).all() & torch.isfinite(blocks).all(),
        "non-finite matrix-free tangent objective",
        FloatingPointError,
    )
    zero = torch.zeros((), device=rhs.device, dtype=torch.float64)
    enabled = torch.ones((), device=rhs.device, dtype=torch.bool)
    norm = dot(rhs, rhs).sqrt()
    inner_tolerance = rtol * 0.25
    requested_tolerance = zero + inner_tolerance
    history = (torch.zeros_like(rhs), torch.zeros_like(rhs), zero.clone())
    previous_shift = zero - 1
    previous_ok = enabled.clone()

    def linear(shift, initial, active, interior=False):
        nonlocal history, previous_shift, previous_ok
        reuse = (
            (shift == previous_shift)
            & ~previous_ok
            & (history[2] > 0)
            & (dot(history[1], history[1]) > 0)
        )
        if rhs.is_cuda and getattr(action, "triton", False):
            p = action.prepared
            result = runner(
                action.rate, action.eps, radius, max_iter, inner_tolerance, interior
            )(
                p._point,
                p._u,
                p._v,
                p._du,
                p._dv,
                action.row,
                action.column,
                rhs,
                blocks,
                shift,
                initial,
                active,
                *history,
                reuse,
                requested_tolerance,
            )
        else:
            result = linear_solve(
                action,
                rhs,
                blocks,
                shift,
                initial,
                active,
                *history,
                reuse,
                radius=radius,
                max_iter=max_iter,
                tolerance=requested_tolerance,
                interior=interior,
            )
        history, previous_shift, previous_ok = result[5:], shift, result[3]
        return result[:5]

    certify = _compiled(certificate) if rhs.is_cuda else certificate
    x, residual, iterations, ok, evaluations = linear(
        zero, torch.zeros_like(rhs), enabled, True
    )
    d, relative, comp, done = certify(x, residual, rhs, zero, radius, rtol)
    done = done & ok
    best = torch.where(done, d, 0)
    best_shift = zero.clone()
    low, high = zero.clone(), norm / radius
    x = torch.where(torch.isfinite(x), x, 0)
    shifts = torch.zeros((), device=rhs.device, dtype=torch.int32)
    for _ in range(shift_steps):
        if not rhs.is_cuda and bool(done):
            break
        shift = (low + high) * 0.5
        x, residual, count, ok, calls = linear(shift, x, ~done)
        evaluations = evaluations + calls
        iterations = iterations + count
        shifts = shifts + (~done).to(shifts.dtype)
        d, relative, comp, valid = certify(x, residual, rhs, shift, radius, rtol)
        accept = ~done & ok & valid
        best = torch.where(accept, d, best)
        best_shift = torch.where(accept, shift, best_shift)
        length = dot(x, x).sqrt()
        # H is PSD, so ||x - x_exact|| <= ||residual|| / shift. Even an
        # inexact solve can certify which side of the radius contains the exact
        # solution. An ambiguous interval stays at this shift for more work.
        error_bound = dot(residual, residual).sqrt() / shift.clamp_min(1e-300)
        outside, inside = length - error_bound > radius, length + error_bound < radius
        low = torch.where(~done & outside, shift, low)
        high = torch.where(~done & inside, shift, high)
        # A converged inner solve can still be too coarse to determine the
        # radius side. Tighten it rather than returning the same iterate forever.
        ambiguous = ~done & ok & ~accept & ~outside & ~inside
        requested_tolerance = torch.where(
            ambiguous,
            (requested_tolerance * 0.1).clamp_min(32 * torch.finfo(rhs.dtype).eps),
            torch.where(outside | inside, zero + inner_tolerance, requested_tolerance),
        )
        done = done | accept
    # Cast first, then independently certify the exact displacement to be committed.
    d = best.to(problem.linear)
    d = d * (radius / d.double().norm().clamp_min(1e-300)).clamp(max=1).to(d.dtype)
    hd = action(d.double())
    residual = hd + best_shift * d - rhs
    _, relative, comp, valid = certify(
        d.double(), residual, rhs, best_shift, radius, rtol
    )
    valid = valid & done
    require(
        valid,
        "matrix-free tangent solver did not meet KKT tolerance",
        FloatingPointError,
    )
    objective = (problem.linear.double() * d).sum() + 0.5 * (d * hd).sum()
    return TangentPCGResult(
        d,
        objective,
        dot(residual, residual).sqrt(),
        0,
        iterations,
        evaluations + 1,
        valid,
        d.double().norm() >= radius * (1 - rtol),
        best_shift,
        relative,
        comp,
        shifts,
    )
