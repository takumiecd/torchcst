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


def output_size(case):
    return case.size if case.fixture == "polar_profile_product_square_strip" else OUTPUT


def fixture_state(case):
    return product_state(case)


def fixture_operator(case):
    out = output_size(case)
    return OperatorSpec(
        layout=SingleChartSpec(
            chart=chart_presets.strip(
                (out, case.size),
                (out, TILE),
                axes=(
                    pattern_presets.line(out, low=0, high=out - 1),
                    pattern_presets.line(case.size, low=0, high=case.size - 1),
                ),
                axis=1,
                tile_pitch=PITCH,
            )
        ),
        kernel=fixture_state(case).spec,
    )


def initialize(case):
    p = product_initialize(
        replace(
            case,
            fixture="polar_profile_product_global"
            if case.fixture == "polar_profile_product_square_strip"
            else "polar_profile_product",
            size=output_size(case),
        )
    )
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
    from .support_report import summarize_axes

    i, o = positions(chart, device=q.device, dtype=q.dtype)
    return {
        "normalization": "whole Strip product L2; one global floor",
        **summarize_axes(q, i, o),
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
    from benchmarks.cuda.linear.protocol import (
        SQUARE_STRIP_PRODUCT_ORACLE_SCOPE,
        STRIP_PRODUCT_ORACLE_SCOPE,
    )
    from benchmarks.cuda.polar_update import optimizer_step

    case = run.case
    p = initialize(case).cuda()
    model = model_type(p, fixture_operator(case), run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, output_size(case), generator=gen).cuda()
    oracle = oracle_vjp
    if case.fixture == "polar_profile_product_square_strip":
        oracle = square_oracle_vjp
    truth, tx, tp = oracle(
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
        "scope": SQUARE_STRIP_PRODUCT_ORACLE_SCOPE
        if case.fixture == "polar_profile_product_square_strip"
        else STRIP_PRODUCT_ORACLE_SCOPE,
        "y": check(y, truth, tol=4e-4),
        "dx": check(gx, tx, tol=4e-4),
        "dp": check(gp, tp, tol=4e-4),
        "polar_update": check(model.p, reference.atoms.p, tol=2e-6),
    }


def square_oracle_vjp(value, p, x, dy, chart, *, chunk=512):
    """Enumerate all factor sites in FP64, never truncate an atom's support."""
    from .global_profile_product import oracle_factors

    inputs, _ = positions(chart, device=p.device, dtype=torch.float64)
    tx = x.detach().double().requires_grad_()
    y = tx.new_zeros((len(x), chart.shape[0]))
    dx, parts = torch.zeros_like(tx), []
    for start in range(0, len(p), chunk):
        tp = p[start : start + chunk].detach().double().requires_grad_()
        u, v, scale = oracle_factors(value, tp, chart, input_sites=inputs)
        yy = ((tx @ v.T) * scale[None, :]) @ u
        gx, gp = torch.autograd.grad(yy, (tx, tp), dy.double())
        y += yy.detach()
        dx += gx
        parts.append(gp)
    return y, dx, torch.cat(parts) if parts else p.double()
