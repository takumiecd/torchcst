"""New product fixture and independent full-site oracle for the existing runner."""

import torch

from torchcst import CSTLinear, TriweightSpec, chart_presets, pattern_presets, presets
from torchcst.kernels.state import KernelState
from torchcst.operators.spec import OperatorSpec, SingleChartSpec


def fixture_state(case):
    from .local_product import fixture_state as previous_state

    old = previous_state(case)
    return KernelState(
        presets.polar_profile_product(
            profiles=(TriweightSpec(), TriweightSpec()),
            amplitude_max=1.0,
            bounds=old.spec.parameterization.input_bounds,
            w_c=1e6,
            dormant_expansion_rate=0.02,
        )
    )


def fixture_operator(case, size=None):
    n = case.size if size is None else size
    return OperatorSpec(
        layout=SingleChartSpec(
            chart=chart_presets.product(
                (n, n),
                (pattern_presets.line(n, low=0, high=n - 1),) * 2,
            )
        ),
        kernel=fixture_state(case).spec,
    )


def initialize(case):
    from .local_product import initialize as previous_initialize

    # Preserve the same seeded coordinate values but canonical output/input order.
    return previous_initialize(case)[:, [0, 1, 3, 2]].contiguous()


def decode(value, p):
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar

    amp, alpha = polar._amplitude_and_alpha(value, p[:, :2])
    sigma, _ = polar._bandwidth_sigmas(value, amp, alpha)
    return torch.stack((amp, sigma.reciprocal().square().detach(), p[:, 3], p[:, 2]), 1)


def summarize(q, domain):
    sites = torch.arange(domain.input_size, dtype=q.dtype)
    v = (1 - (sites[:, None] - q[:, 2]).square() * q[:, 1]).clamp_min(0).pow(3)
    u = (1 - (sites[:, None] - q[:, 3]).square() * q[:, 1]).clamp_min(0).pow(3)
    norms = v.norm(dim=0) * u.norm(dim=0)
    return {
        "normalization": "full product L2; one floor 1e-6",
        "empty_atoms": int(
            ((v.count_nonzero(dim=0) == 0) | (u.count_nonzero(dim=0) == 0)).sum()
        ),
        "floor_active_atoms": int((norms < 1e-6).sum()),
        "onehot_both_live_atoms": int(
            (
                (v.count_nonzero(dim=0) == 1)
                & (u.count_nonzero(dim=0) == 1)
                & (norms >= 1e-6)
            ).sum()
        ),
    }


def oracle_atoms(value, p, n, *, input_sites=None, output_sites=None):
    """Hand-written Polar map; enumerate complete matrices and their L2 norm."""
    q = p[:, :2].square().sum(-1)
    amp = value.amplitude_max * p[:, 0] / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    x = (amp / value.w_c).square()
    lower = value.sigma_min_input + (
        value.sigma_birth_input - value.sigma_min_input
    ) / (1 + value.lower_kappa * x)
    upper = value.sigma_min_input + (
        value.sigma_max_input - value.sigma_min_input
    ) * value.kappa / (value.kappa + x.pow(value.upper_decay_power))
    upper = torch.maximum(torch.maximum(upper, value.upper_floor_input), lower)
    sigma = (
        torch.exp((1 - alpha) * lower.log() + alpha * upper.log())
        .clamp(min=lower, max=upper)
        .detach()
    )
    sites = torch.arange(n, device=p.device, dtype=p.dtype)
    output_sites = sites if output_sites is None else output_sites
    input_sites = sites if input_sites is None else input_sites
    u = (
        (1 - (output_sites[None, :] - p[:, 2, None]).square() / sigma[:, None].square())
        .clamp_min(0)
        .pow(3)
    )
    v = (
        (1 - (input_sites[None, :] - p[:, 3, None]).square() / sigma[:, None].square())
        .clamp_min(0)
        .pow(3)
    )
    raw = u[:, :, None] * v[:, None, :]
    norm = torch.linalg.vector_norm(raw.flatten(1), dim=1)
    return raw * (amp / norm.clamp_min(value.spec.normalization.floor))[:, None, None]


def correctness(args, run, model_type):
    from torchcst import CSTOptimizer

    from .check_normalized import check

    case = run.case
    p = initialize(case).cuda()
    model = model_type(p, fixture_operator(case), run.entry(args.plan_id).plan)
    gen = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.rows, case.size, generator=gen).cuda().requires_grad_()
    dy = torch.randn(case.rows, case.size, generator=gen).cuda()
    tx, tp = x.detach().double().requires_grad_(), p.double().requires_grad_()
    value = fixture_state(case).double().cuda()
    truth = tx @ oracle_atoms(value, tp, case.size).sum(0).T
    y = model(x)
    ga = torch.autograd.grad(y, (x, model.p), dy)
    gt = torch.autograd.grad(truth, (tx, tp), dy.double())
    if not torch.all(gt[1][:, 2:].abs() > 1e-8):
        raise AssertionError("performance oracle needs nonzero center derivatives")
    # Shared public update contract, with the exact same task cotangents.
    reference = CSTLinear(
        chart=fixture_operator(case).charts[0],
        atoms=p,
        kernel=value.spec,
        device="cuda",
    )
    opt = CSTOptimizer(
        torch.optim.AdamW(
            reference.parameters(),
            lr=case.optimizer.lr,
            weight_decay=case.optimizer.weight_decay,
            fused=True,
            capturable=False,
        ),
        model=reference,
    )
    reference.atoms.p.grad = ga[1].detach().clone()
    from benchmarks.cuda.polar_update import optimizer_step

    candidate_opt = torch.optim.AdamW(
        model.parameters(),
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        fused=True,
        capturable=True,
    )
    model.p.grad = ga[1].detach().clone()
    optimizer_step(
        model.update_binding,
        candidate_opt,
        step_size=case.optimizer.lr,
        polar_update=args.polar_update,
    )
    opt.step()
    return {
        "status": "PASS",
        "scope": "independent FP64 full-site Y/dX/all atom gradients; production polar update",
        "y": check(y, truth, tol=4e-4),
        "dx": check(ga[0], gt[0], tol=4e-4),
        "dp": check(ga[1], gt[1], tol=4e-4),
        "polar_update": check(model.p, reference.atoms.p, tol=2e-6),
    }
