"""Large Product fixture with a chunked independent FP64 full-site VJP."""

import hashlib
import json

import torch

from .profile_product import fixture_operator, fixture_state, initialize


def compact_width_record(record):
    """Keep full series in the worker JSON; bound the consolidated artifact."""
    result = record["result"]
    sigma = result.get("sigma_updates")
    if sigma is None or "initial" not in sigma:
        return record
    initial, final = sigma["initial"], sigma["final"]
    atoms = result["atoms"]
    if len(initial) != atoms or len(final) != atoms or not atoms:
        raise ValueError("width series must cover every atom")
    compact = {k: v for k, v in sigma.items() if k not in ("initial", "final")}
    compact.update(
        atoms=atoms,
        series_storage="full initial/final arrays in the unmodified measure worker JSON",
        hash_encoding="SHA256 of each exact JSON array with separators comma/colon",
        worker_record=f"measure-{record['metadata']['plan_id']}.json",
        initial_range=[min(initial), max(initial)],
        final_range=[min(final), max(final)],
        initial_sha256=hashlib.sha256(
            json.dumps(initial, separators=(",", ":")).encode()
        ).hexdigest(),
        final_sha256=hashlib.sha256(
            json.dumps(final, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return record | {"result": result | {"sigma_updates": compact}}


def oracle_factors(value, p, chart):
    # Written from the public Polar definition; no CUDA preparation/VJP reuse.
    radius2 = p[:, :2].square().sum(-1)
    amp = (
        value.amplitude_max
        * p[:, 0]
        / radius2.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    )
    alpha = ((radius2 - 1) / 3).clamp(0, 1)
    activity = (amp / value.w_c).square()
    lower = value.sigma_min_input + (
        value.sigma_birth_input - value.sigma_min_input
    ) / (1 + value.lower_kappa * activity)
    upper = value.sigma_min_input + (
        value.sigma_max_input - value.sigma_min_input
    ) * value.kappa / (value.kappa + activity.pow(value.upper_decay_power))
    upper = torch.maximum(torch.maximum(upper, value.upper_floor_input), lower)
    sigma = (
        torch.exp((1 - alpha) * lower.log() + alpha * upper.log())
        .clamp(min=lower, max=upper)
        .detach()
    )
    out = (
        torch.arange(chart.shape[0], device=p.device, dtype=p.dtype)
        * chart.axes[0].spacing[0]
        + chart.axes[0].start[0]
    )
    inp = (
        torch.arange(chart.shape[1], device=p.device, dtype=p.dtype)
        * chart.axes[1].spacing[0]
        + chart.axes[1].start[0]
    )
    u = (
        (1 - (out[None, :] - p[:, 2, None]).square() / sigma[:, None].square())
        .clamp_min(0)
        .pow(3)
    )
    v = (
        (1 - (inp[None, :] - p[:, 3, None]).square() / sigma[:, None].square())
        .clamp_min(0)
        .pow(3)
    )
    denominator = (u.norm(dim=1) * v.norm(dim=1)).clamp_min(
        value.spec.normalization.floor
    )
    return u, v, amp / denominator


def oracle_vjp(value, p, x, dy, chart, chunk=512):
    tx = x.detach().double().requires_grad_()
    y, dx, grads = tx.new_zeros((len(x), chart.shape[0])), torch.zeros_like(tx), []
    for start in range(0, len(p), chunk):
        tp = p[start : start + chunk].detach().double().requires_grad_()
        u, v, scale = oracle_factors(value, tp, chart)
        yy = ((tx @ v.T) * scale[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, tp), dy.double())
        y += yy.detach()
        dx += gx
        grads.append(gp)
    return y, dx, torch.cat(grads) if grads else p.double()


def summarize(q, chart):
    result = {
        "normalization": "whole Product L2; one floor1e-6",
        "empty_atoms": 0,
        "floor_active_atoms": 0,
        "onehot_both_live_atoms": 0,
    }
    out = (
        torch.arange(chart.shape[0], dtype=q.dtype) * chart.axes[0].spacing[0]
        + chart.axes[0].start[0]
    )
    inp = (
        torch.arange(chart.shape[1], dtype=q.dtype) * chart.axes[1].spacing[0]
        + chart.axes[1].start[0]
    )
    for part in q.split(1024):
        v = (1 - (inp[:, None] - part[:, 2]).square() * part[:, 1]).clamp_min(0).pow(3)
        u = (1 - (out[:, None] - part[:, 3]).square() * part[:, 1]).clamp_min(0).pow(3)
        norm, nv, nu = (
            v.norm(dim=0) * u.norm(dim=0),
            v.count_nonzero(dim=0),
            u.count_nonzero(dim=0),
        )
        result["empty_atoms"] += int(((nv == 0) | (nu == 0)).sum())
        result["floor_active_atoms"] += int((norm < 1e-6).sum())
        result["onehot_both_live_atoms"] += int(
            ((nv == 1) & (nu == 1) & (norm >= 1e-6)).sum()
        )
    return result


def correctness(args, run, model_type):
    from benchmarks.cuda.polar_update import optimizer_step
    from torchcst import CSTLinear, CSTOptimizer

    from .check_normalized import check
    from .protocol import GLOBAL_PRODUCT_ORACLE_SCOPE

    case = run.case
    p = initialize(case).cuda()
    op = fixture_operator(case)
    model = model_type(p, op, run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, case.size, generator=gen).cuda()
    truth, tx, tp = oracle_vjp(
        fixture_state(case).double().cuda(), p, x, dy, op.charts[0]
    )
    y = model(x)
    ga = torch.autograd.grad(y, (x, model.p), dy)
    reference = CSTLinear(
        chart=op.charts[0], atoms=p.clone(), kernel=op.kernel, device="cuda"
    )
    opt = CSTOptimizer(
        torch.optim.AdamW(
            reference.parameters(),
            lr=case.optimizer.lr,
            weight_decay=case.optimizer.weight_decay,
            fused=True,
        ),
        model=reference,
    )
    reference.atoms.p.grad = ga[1].detach().clone()
    candidate = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
        capturable=True,
    )
    model.p.grad = ga[1].detach().clone()
    optimizer_step(
        model.update_binding,
        candidate,
        step_size=case.optimizer.lr,
        polar_update=args.polar_update,
    )
    opt.step()
    return {
        "status": "PASS",
        "scope": GLOBAL_PRODUCT_ORACLE_SCOPE,
        "y": check(y, truth, tol=4e-4),
        "dx": check(ga[0], tx, tol=4e-4),
        "dp": check(ga[1], tp, tol=4e-4),
        "polar_update": check(model.p, reference.atoms.p, tol=2e-6),
    }
