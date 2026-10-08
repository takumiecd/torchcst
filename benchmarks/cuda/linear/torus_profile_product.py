"""Square Strip/S1xS2 fixture and independent embedded-fibre FP64 oracle."""

import copy
import math

import torch

from torchcst import (
    CSTLinear,
    CSTOptimizer,
    TriweightSpec,
    chart_presets,
    geometry_presets,
    pattern_presets,
    presets,
)
from torchcst.kernels.state import KernelState
from torchcst.operators.spec import OperatorSpec, SingleChartSpec

from .local_product import (
    fixture_state as previous_state,
)
from .local_product import (
    initialize as previous_initialize,
)


def fixture_state(case):
    return KernelState(
        presets.polar_torus_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1.0,
            bounds=previous_state(case).spec.parameterization.input_bounds,
            w_c=1e6,
            dormant_expansion_rate=0.02,
        )
    )


def fixture_operator(case):
    n = case.size
    rows, columns = 32, n // 32
    return OperatorSpec(
        layout=SingleChartSpec(
            chart=chart_presets.strip(
                (n, n),
                (64, n),
                axis=0,
                tile_pitch=68.0,
                axes=(
                    pattern_presets.line(n, low=0, high=n - 1),
                    pattern_presets.grid(
                        (rows, columns),
                        low=(-(rows - 1) / 2, -(columns - 1) / 2),
                        high=((rows - 1) / 2, (columns - 1) / 2),
                    ),
                ),
                geometry=geometry_presets.torus(
                    3,
                    major_radius=math.ceil(n / 64) * 68 / (2 * math.pi),
                    minor_radius=math.sqrt(n),
                    representation="intrinsic",
                    max_arc_step=0.2,
                ),
            )
        ),
        kernel=fixture_state(case).spec,
    )


def initialize(case):
    polar = previous_initialize(case)[:, :2]
    n, columns, minor = case.size, case.size // 32, math.sqrt(case.size)
    gen = torch.Generator().manual_seed(case.seed + 907)
    indices = torch.randint(n, (case.atoms, 2), generator=gen)
    arc = (indices[:, 0] % 64).float() + (indices[:, 0] // 64).float() * 68 + 0.37
    period = math.ceil(n / 64) * 68
    arc = (arc + period / 2).remainder(period) - period / 2
    cross = torch.stack(
        (
            (indices[:, 1] // columns).float() - 15.5 + 0.37,
            (indices[:, 1] % columns).float() - (columns - 1) / 2 + 0.37,
        ),
        -1,
    )
    length = cross.norm(dim=-1, keepdim=True)
    section = minor * (length / minor).atan() * cross / length.clamp_min(1e-20)
    return torch.cat((polar, arc[:, None], section), -1).contiguous()


def decode(value, p):
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar

    amp, alpha = polar._amplitude_and_alpha(value, p[:, :2])
    sigma, _ = polar._bandwidth_sigmas(value, amp, alpha)
    return torch.cat(
        (amp[:, None], sigma.reciprocal().square().detach()[:, None], p[:, 2:]), -1
    )


def _coordinates(chart, p):
    # Independent chart enumeration, including Strip's physical gaps.
    j = torch.arange(chart.shape[0], device=p.device)
    circle = chart.axes[0]
    arc = circle.start[0].to(p) + (j % chart.tile_shape[0]).to(
        p.dtype
    ) * circle.spacing[0].to(p)
    arc = arc + (j // chart.tile_shape[0]).to(p.dtype) * chart.tile_pitch.to(p)
    grid = chart.axes[1]
    vectors = [
        grid.start[k].to(p)
        + torch.arange(s, device=p.device, dtype=p.dtype) * grid.spacing[k].to(p)
        for k, s in enumerate(grid.spec.shape)
    ]
    cross = torch.stack(torch.meshgrid(*vectors, indexing="ij"), -1).reshape(-1, 2)
    major, minor = chart.geometry.major_radius.to(p), chart.geometry.minor_radius.to(p)
    section = torch.cat((minor.expand(len(cross), 1), cross), -1)
    section = section / section.norm(dim=-1, keepdim=True)
    return arc / major, section, major, minor


def _polar(value, p):
    # Hand-written task map; no kernel/parameterization backend calls.
    r2 = p[:, :2].square().sum(-1)
    amp = value.amplitude_max * p[:, 0] / r2.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((r2 - 1) / 3).clamp(0, 1)
    z = (amp / value.w_c).square()
    lo = value.sigma_min_input + (value.sigma_birth_input - value.sigma_min_input) / (
        1 + value.lower_kappa * z
    )
    hi = value.sigma_min_input + (
        value.sigma_max_input - value.sigma_min_input
    ) * value.kappa / (value.kappa + z.pow(value.upper_decay_power))
    hi = torch.maximum(torch.maximum(hi, value.upper_floor_input), lo)
    sigma = (
        torch.exp((1 - alpha) * lo.log() + alpha * hi.log())
        .clamp(min=lo, max=hi)
        .detach()
    )
    return amp, sigma


def oracle_factors(value, p, chart):
    """Actual embedded fibre queries, not the runtime's distance formulas."""
    angles, sites, major, minor = _coordinates(chart, p)
    section_angle = p[:, 3:].norm(dim=-1, keepdim=True) / minor
    q = torch.cat(
        (section_angle.cos(), torch.sinc(section_angle / torch.pi) * p[:, 3:] / minor),
        -1,
    )
    theta = p[:, 2] / major

    def embed(t, qq):
        radius = major + minor * qq[..., 0]
        return torch.cat(
            (
                torch.stack((radius * t.cos(), radius * t.sin()), -1),
                minor * qq[..., 1:],
            ),
            -1,
        )

    centre = embed(theta, q)
    circle_query = embed(
        angles[:, None].expand(-1, len(p)), q[None].expand(len(angles), -1, -1)
    )
    section_query = embed(
        theta[None].expand(len(sites), -1), sites[:, None].expand(-1, len(p), -1)
    )
    amp, sigma = _polar(value, p)
    raw = [
        (1 - (query - centre[None]).square().sum(-1) / sigma[None].square())
        .clamp_min(0)
        .pow(3)
        for query in (circle_query, section_query)
    ]
    u, v = raw
    denominator = (u.norm(dim=0) * v.norm(dim=0)).clamp_min(
        value.spec.normalization.floor
    )
    return u, v, amp / denominator


def oracle_vjp(value, p, x, dy, chart, chunk=256):
    tx = x.detach().double().requires_grad_()
    y, dx, grads = tx.new_zeros((len(x), chart.shape[0])), torch.zeros_like(tx), []
    chart = copy.deepcopy(chart).double()
    for start in range(0, len(p), chunk):
        tp = p[start : start + chunk].detach().double().requires_grad_()
        u, v, scale = oracle_factors(value, tp, chart)
        yy = ((tx @ v) * scale[None]) @ u.T
        gx, gp = torch.autograd.grad(yy, (tx, tp), dy.double())
        y += yy.detach()
        dx += gx
        grads.append(gp)
    return y, dx, torch.cat(grads) if grads else p.double()


def summarize(q, chart):
    """Untimed CPU full-axis support accounting, with bounded atom blocks."""
    from torchcst import compile_chart

    chart = compile_chart(chart, dtype=q.dtype, device=q.device)
    angles, sites, major, minor = _coordinates(chart, q)
    counts = {"empty_atoms": 0, "floor_active_atoms": 0, "onehot_both_live_atoms": 0}
    for start in range(0, len(q), 256):
        z = q[start : start + 256]
        angle = z[:, 3:].norm(dim=-1, keepdim=True) / minor
        centre = torch.cat(
            (angle.cos(), torch.sinc(angle / torch.pi) * z[:, 3:] / minor), -1
        )
        radius = major + minor * centre[:, 0]
        dc = (
            4
            * radius[None].square()
            * ((angles[:, None] - z[None, :, 2] / major) / 2).sin().square()
        )
        ds = minor.square() * (sites[:, None] - centre[None]).square().sum(-1)
        u, v = [(1 - d * z[None, :, 1]).clamp_min(0).pow(3) for d in (dc, ds)]
        nu, nv = u.count_nonzero(dim=0), v.count_nonzero(dim=0)
        norm = u.norm(dim=0) * v.norm(dim=0)
        counts["empty_atoms"] += int(((nu == 0) | (nv == 0)).sum())
        counts["floor_active_atoms"] += int((norm < 1e-6).sum())
        counts["onehot_both_live_atoms"] += int(
            ((nu == 1) & (nv == 1) & (norm >= 1e-6)).sum()
        )
    return {
        "normalization": "complete S1xS2 Strip; one operator L2 floor1e-6",
        **counts,
    }


def correctness(args, run, model_type):
    from benchmarks.cuda.polar_update import optimizer_step

    from .check_normalized import check
    from .protocol import TORUS_PRODUCT_ORACLE_SCOPE

    case, op = run.case, fixture_operator(run.case)
    p = initialize(case).cuda()
    model = model_type(p, op, run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, case.size, generator=gen).cuda()
    truth, tx, tp = oracle_vjp(
        copy.deepcopy(model.local_state).double(), p, x, dy, model.product_site.chart
    )
    y = model(x)
    gx, gp = torch.autograd.grad(y, (x, model.p), dy)
    # A nontrivial fixture, rather than passing all-zero task comparisons.
    if not bool((tp[:, 2:].abs() > 1e-8).all()):
        raise AssertionError("performance fixture needs nonzero centre gradients")
    ref = CSTLinear(
        chart=op.charts[0], atoms=p.clone(), kernel=op.kernel, device="cuda"
    )
    public = CSTOptimizer(
        torch.optim.AdamW(
            ref.parameters(),
            lr=case.optimizer.lr,
            weight_decay=case.optimizer.weight_decay,
            fused=True,
        ),
        model=ref,
    )
    candidate = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
        capturable=True,
    )
    ref.atoms.p.grad, model.p.grad = gp.clone(), gp.clone()
    public.step()
    optimizer_step(
        model.update_binding,
        candidate,
        step_size=case.optimizer.lr,
        polar_update=args.polar_update,
    )
    return {
        "status": "PASS",
        "scope": TORUS_PRODUCT_ORACLE_SCOPE,
        "y": check(y, truth, tol=4e-4),
        "dx": check(gx, tx, tol=4e-4),
        "dp": check(gp, tp, tol=4e-4),
        "polar_update": check(model.p, ref.atoms.p, tol=2e-6),
    }
