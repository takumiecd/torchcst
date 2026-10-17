"""Quadratic B-spline fixtures and independent FP64 full-axis oracle."""

import copy
import math

import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    QuadraticBSplineSpec,
    chart_presets,
    presets,
)
from torchcst.kernels.state import KernelState
from torchcst.operators.spec import OperatorSpec, SingleChartSpec


def fixture_state(case):
    w = case.widths
    bounds = BandwidthBounds(
        minimum=w.minimum if w else 0.1,
        birth=w.birth if w else 0.1,
        maximum=w.maximum if w else 12.8,
        upper_floor=w.minimum if w else 0.1,
    )
    return KernelState(
        presets.polar_profile_product(
            profiles=(QuadraticBSplineSpec(),) * 2,
            amplitude_max=1.0,
            bounds=bounds,
            w_c=1e6,
            dormant_expansion_rate=0.02,
        )
    )


def fixture_operator(case, size=None):
    n = case.size if size is None else size
    return OperatorSpec(
        layout=SingleChartSpec(
            chart=chart_presets.regular_grid(((n,), (n,)), spacing=0.1)
        ),
        kernel=fixture_state(case).spec,
    )


def initialize(case):
    gen = torch.Generator().manual_seed(case.seed)
    a = case.atoms
    direction = torch.rand(a, generator=gen) * 1.2 - 0.6
    if case.widths:
        w = case.widths
        ends = [round(a * sum(w.fractions[: k + 1])) for k in range(len(w.rho))]
        sigma = torch.empty(a)
        begin = 0
        for s, end in zip(w.rho, ends):
            sigma[begin:end] = s
            begin = end
        sigma = sigma[torch.randperm(a, generator=gen)]
        lo, hi = w.birth, w.maximum
    else:
        lo, hi = 0.1, 12.8
        sigma = torch.exp(
            torch.rand(a, generator=gen) * (math.log(hi) - math.log(lo)) + math.log(lo)
        )
    alpha = (sigma / lo).log() / math.log(hi / lo)
    radius = (1 + 3 * alpha).sqrt()
    return torch.cat(
        (
            torch.stack((direction, (1 - direction.square()).sqrt()), 1)
            * radius[:, None],
            torch.rand(a, 2, generator=gen) * (case.size - 1) * 0.1,
        ),
        1,
    )


def decode(value, p):
    from torchcst._backends.torch.parameterizations.polar_amp_width import (
        _amplitude_and_alpha,
        _sigma_bounds,
    )

    amp, alpha = _amplitude_and_alpha(
        value, p[:, :2].double(), tiny=torch.finfo(p.dtype).tiny
    )
    sigma = _sigma_bounds(value, amp, alpha)[0]
    return torch.stack((amp, sigma.detach().reciprocal().square(), p[:, 3], p[:, 2]), 1)


def summarize(q, domain):
    return {
        "normalization": "complete sampled product L2, floor once",
        "atoms": len(q),
        "sigma_min": float(q[:, 1].rsqrt().min()),
        "sigma_max": float(q[:, 1].rsqrt().max()),
    }


def oracle(value, chart, p, x):
    # Independent signed polynomial pieces; full axis samples and direct norms.
    # The Polar decode is explicitly written in the established independent oracle.
    # Obtain amplitudes/widths independently, without its Triweight shape.
    r2 = p[:, :2].square().sum(-1)
    amp = (
        value.amplitude_max
        * p[:, 0]
        / r2.clamp_min(torch.finfo(torch.float32).tiny).sqrt()
    )
    alpha = ((r2 - 1) / 3).clamp(0, 1)
    u = (amp / value.w_c).square()
    low = value.sigma_min_input + (value.sigma_birth_input - value.sigma_min_input) / (
        1 + value.lower_kappa * u
    )
    high = value.sigma_min_input + (
        value.sigma_max_input - value.sigma_min_input
    ) * value.kappa / (value.kappa + u.pow(value.upper_decay_power))
    high = torch.maximum(torch.maximum(high, value.upper_floor_input), low)
    sigma = (
        ((1 - alpha) * low.log() + alpha * high.log())
        .exp()
        .clamp(min=low, max=high)
        .detach()
    )
    axes = []
    for dim, n in enumerate((chart.grid_shape[0][0], chart.grid_shape[1][0])):
        sites = (
            chart.origin[dim]
            + torch.arange(n, device=p.device, dtype=p.dtype) * chart.spacing[dim]
        )
        t = (sites[:, None] - p[None, :, dim + 2]) / (math.sqrt(8 / 3) * sigma[None, :])
        f = torch.where(
            t < -0.5,
            (2 / 3) * (t + 1.5).square(),
            torch.where(
                t <= 0.5, 1 - (4 / 3) * t.square(), (2 / 3) * (1.5 - t).square()
            ),
        )
        axes.append(torch.where(t.abs() < 1.5, f, torch.zeros_like(f)))
    fo, fi = axes
    denom = (fo.norm(dim=0) * fi.norm(dim=0)).clamp_min(value.spec.normalization.floor)
    return ((x @ fi) * (amp / denom)[None, :]) @ fo.T


def correctness(args, run, model_type):
    from .check_normalized import check

    case = run.case
    model = model_type(
        initialize(case).cuda(), fixture_operator(case), run.entry(args.plan_id).plan
    )
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn_like(x)
    reports = []
    for step in range(3):
        if step:
            with torch.no_grad():
                model.p[:, :2].mul_(1.001)
                model.p[:, 2:].add_(0.0037)
        tx = x.detach().double().requires_grad_()
        tp = model.p.detach().double().requires_grad_()
        truth = oracle(
            copy.deepcopy(model.local_state).double(), model.product_site.chart, tp, tx
        )
        # Keep the live kernel scalar arithmetic at its original FP32 contract.
        y = model(x)
        ga = torch.autograd.grad(y, (x, model.p), dy, retain_graph=True)
        gt = torch.autograd.grad(truth, (tx, tp), dy.double())
        reports.append(
            {
                "step": step,
                "y": check(y, truth, 4e-4),
                "dx": check(ga[0], gt[0], 4e-4),
                "dp": check(ga[1], gt[1], 4e-4),
            }
        )
        repeated = torch.autograd.grad(y, (x, model.p), dy)
        for a, b in zip(ga, repeated):
            check(a, b, 4e-4)
    from benchmarks.cuda.polar_update import optimizer_step

    reference = CSTLinear(
        chart=fixture_operator(case).charts[0],
        atoms=model.p.detach().clone(),
        kernel=model.local_state.spec,
        backend="factored",
        device="cuda",
    )
    reference.chart.double()
    reference.chart.spacing.fill_(0.1)
    reference.atoms.p.grad = ga[1].detach().clone()
    model.p.grad = ga[1].detach().clone()
    kwargs = {
        "lr": case.optimizer.lr,
        "weight_decay": case.optimizer.weight_decay,
        "fused": True,
    }
    public = CSTOptimizer(
        torch.optim.AdamW(reference.parameters(), capturable=False, **kwargs),
        model=reference,
    )
    candidate = torch.optim.AdamW(model.parameters(), capturable=True, **kwargs)
    public.step()
    optimizer_step(
        model.update_binding,
        candidate,
        step_size=case.optimizer.lr,
        polar_update=args.polar_update,
    )
    update = check(model.p, reference.atoms.p, 4e-4)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        result = check(
            candidate.state[model.p][name],
            public.base_optimizer.state[reference.atoms.p][name],
            4e-4,
        )
        update = {key: max(update[key], result[key]) for key in update}
    assert all(
        r[name][key] < 4e-4
        for r in reports
        for name in ("y", "dx", "dp")
        for key in ("max", "rel_l2")
    )
    return {
        "status": "PASS",
        "scope": "independent FP64 full-axis Y/dX/all atom gradients; three live states",
        **{
            name: {key: max(r[name][key] for r in reports) for key in ("max", "rel_l2")}
            for name in ("y", "dx", "dp")
        },
        "polar_update": update,
        "states": reports,
    }
