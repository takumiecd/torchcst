"""Input-strip fixture and independent complete-site chunked FP64 oracle."""

from dataclasses import replace

import torch

from torchcst import CSTLinear, CSTOptimizer, chart_presets, pattern_presets
from torchcst.operators.spec import OperatorSpec, SingleChartSpec

from .profile_product import (
    fixture_state as product_state,
)
from .profile_product import (
    initialize as product_initialize,
)
from .profile_product import oracle_atoms

TILE = 64
PITCH = 68.0
OUTPUT = 64


def fixture_state(case):
    return product_state(case)


def fixture_operator(case):
    return OperatorSpec(
        layout=SingleChartSpec(
            chart=chart_presets.strip(
                (OUTPUT, case.size),
                (OUTPUT, TILE),
                axes=(
                    pattern_presets.line(OUTPUT, low=0, high=OUTPUT - 1),
                    pattern_presets.line(case.size, low=0, high=case.size - 1),
                ),
                axis=1,
                tile_pitch=PITCH,
            )
        ),
        kernel=fixture_state(case).spec,
    )


def initialize(case):
    p = product_initialize(replace(case, fixture="polar_profile_product", size=64))
    gen = torch.Generator().manual_seed(case.seed + 907)
    logical = torch.randint(0, case.size, (case.atoms,), generator=gen)
    p[:, 3] = (logical % TILE).float() + (logical // TILE).float() * PITCH + 0.37
    return p


def positions(chart, *, device, dtype, pitch=None):
    j = torch.arange(chart.shape[1], device=device)
    tile = chart.tile_shape[1]
    inputs = (
        chart.axes[1].start[0]
        + (j % tile).to(dtype) * chart.axes[1].spacing[0]
        + (j // tile).to(dtype) * (chart.tile_pitch if pitch is None else pitch)
    )
    outputs = (
        chart.axes[0].start[0]
        + torch.arange(chart.shape[0], device=device, dtype=dtype)
        * chart.axes[0].spacing[0]
    )
    return inputs, outputs


def summarize(q, chart):
    i, o = positions(chart, device=q.device, dtype=q.dtype)
    v = (1 - (i[:, None] - q[:, 2]).square() * q[:, 1]).clamp_min(0).pow(3)
    u = (1 - (o[:, None] - q[:, 3]).square() * q[:, 1]).clamp_min(0).pow(3)
    norm = v.norm(dim=0) * u.norm(dim=0)
    return {
        "normalization": "whole Strip product L2; one global floor",
        "empty_atoms": int(
            ((v.count_nonzero(dim=0) == 0) | (u.count_nonzero(dim=0) == 0)).sum()
        ),
        "floor_active_atoms": int((norm < 1e-6).sum()),
        "onehot_both_live_atoms": int(
            (
                (v.count_nonzero(dim=0) == 1)
                & (u.count_nonzero(dim=0) == 1)
                & (norm >= 1e-6)
            ).sum()
        ),
        "input_tiles": (chart.shape[1] + chart.tile_shape[1] - 1)
        // chart.tile_shape[1],
    }


def oracle_vjp(value, p, x, dy, chart, *, pitch=None, chunk=32):
    i, o = positions(chart, device=p.device, dtype=torch.float64, pitch=pitch)
    tx = x.detach().double().requires_grad_()
    y = tx.new_zeros((len(x), chart.shape[0]))
    dx = torch.zeros_like(tx)
    parts = []
    for start in range(0, len(p), chunk):
        tp = p[start : start + chunk].detach().double().requires_grad_()
        matrices = oracle_atoms(value, tp, 1, input_sites=i, output_sites=o)
        yy = tx @ matrices.sum(0).T
        gx, gp = torch.autograd.grad(yy, (tx, tp), dy.double())
        y += yy.detach()
        dx += gx
        parts.append(gp)
    return y, dx, torch.cat(parts) if parts else p.double()


def correctness(args, run, model_type):
    import copy

    from benchmarks.cuda.linear.check_normalized import check
    from benchmarks.cuda.linear.protocol import STRIP_PRODUCT_ORACLE_SCOPE
    from benchmarks.cuda.polar_update import optimizer_step

    case = run.case
    p = initialize(case).cuda()
    model = model_type(p, fixture_operator(case), run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, OUTPUT, generator=gen).cuda()
    truth, tx, tp = oracle_vjp(
        copy.deepcopy(model.local_state).double(),
        p,
        x,
        dy,
        fixture_operator(case).charts[0],
    )
    if not bool((tp[:, 2:].abs() > 1e-8).all()):
        raise AssertionError("performance fixture must have nonzero center gradients")
    y = model(x)
    gx, gp = torch.autograd.grad(y, (x, model.p), dy)
    reference = CSTLinear(
        chart=fixture_operator(case).charts[0],
        atoms=p.clone(),
        kernel=fixture_state(case).spec,
        device="cuda",
    )
    base = torch.optim.AdamW(
        reference.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
    )
    public = CSTOptimizer(base, model=reference)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
        capturable=True,
    )
    reference.atoms.p.grad = gp.clone()
    model.p.grad = gp.clone()
    public.step()
    optimizer_step(
        model.update_binding,
        optimizer,
        step_size=case.optimizer.lr,
        polar_update=args.polar_update,
    )
    return {
        "status": "PASS",
        "scope": STRIP_PRODUCT_ORACLE_SCOPE,
        "y": check(y, truth, tol=4e-4),
        "dx": check(gx, tx, tol=4e-4),
        "dp": check(gp, tp, tol=4e-4),
        "polar_update": check(model.p, reference.atoms.p, tol=2e-6),
    }
