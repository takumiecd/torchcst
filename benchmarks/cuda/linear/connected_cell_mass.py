"""Connected profile fixture/adapter for the existing Linear measurement runner.

This is a new mathematical fixture, never projected onto legacy radius results.
The same physical Atom rows and AdamW update are used by all three CST plans.
"""

import gc
import math
import statistics
import time
from dataclasses import asdict, replace

import torch

from torchcst import ChebyshevSpec, CSTLinear, chart_presets, kernels
from torchcst.kernels.spec import RadiusConnectionSpec

from .radius_cell_mass import error, tensor_hash


def kernel(modes, connected):
    spec = kernels.product(
        kernels.cell_mass(ChebyshevSpec(modes=modes)),
        coordinates=kernels.physical(radius_init=(0.5, 32.0)),
    )
    if connected:
        spec = replace(
            spec,
            revision=9,
            profiles=tuple(
                replace(p, connection=RadiusConnectionSpec()) for p in spec.profiles
            ),
        )
    return spec


def operator(size, modes, connected):
    chart = chart_presets.regular_grid(size, spacing=1.0, geometry="flat_torus")
    model = CSTLinear(
        chart, chart, atoms=1, kernel=kernel(modes, connected), backend="factored"
    )
    return model.execution_declaration()


def parameters(size, atoms, seed):
    gen = torch.Generator().manual_seed(seed)
    p = torch.empty(atoms, 4)
    p[:, 0] = torch.randn(atoms, generator=gen) * 0.1
    # Deliberately spans small direct tiles and wide shared-moment banks.
    p[:, 1] = (0.5 * (64 ** torch.rand(atoms, generator=gen))).clamp_max(
        min(32, size / 2)
    )
    p[:, 2:] = torch.rand(atoms, 2, generator=gen) * size
    return p


def scalar_axis(row, size, modes, connected, column):
    """Independent FP64 quadrature of T_M(beta*cos)/T_M(beta), no Fourier FFT."""

    import numpy as np

    r, c = row[1], row[column].remainder(size)
    t = 1
    while float(r.detach()) > ((0.9 if connected else 1) * t):
        t *= 2

    def s(x):
        x = x.clamp(0, 1)
        return x**3 * (10 - 15 * x + 6 * x * x)

    scales = [(t, r * 0 + 1)]
    if connected:
        w = s(torch.log(r / (0.72 * t)) / math.log(1.25))
        scales = [(t, 1 - w), (t * 2, w)]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes, weights = row.new_tensor(nodes), row.new_tensor(weights)
    result = row.new_zeros(size)
    for t, sw in scales:
        if float(sw.detach()) == 0:
            continue
        k = 3 if connected else math.ceil(2 * float(r.detach()) / t) + 1
        period = k * t
        q = (c + 0.5) / t - ((k - 1) / 2 if not connected else 1.5)
        n = int(q.detach().floor())
        windows = [(n, c * 0 + 1)]
        if connected:
            pw = s((q - n - 0.4) / 0.2)
            windows = [(n, 1 - pw), (n + 1, pw)]

        def integ(left, right, period=period):
            u = (right - left)[..., None] / 2 * nodes + (right + left)[..., None] / 2
            beta = 1 / torch.cos(torch.pi * r / period)
            v = beta * torch.cos(torch.pi * u / period)
            a, b = torch.ones_like(v), v
            for _ in range(2, modes + 1):
                a, b = b, 2 * v * b - a
            norm = torch.cosh(modes * torch.acosh(beta))
            return (right - left) / 2 * ((b / norm) ** 2 * weights).sum(-1)

        norm = integ(row.new_tensor(-period / 2), row.new_tensor(period / 2))
        for n, wp in windows:
            if float(wp.detach()) == 0:
                continue
            sites = torch.arange(n * t, (n + k) * t, dtype=row.dtype)
            left = sites - 0.5 - c
            vals = integ(left, left + 1) / norm
            result = result.scatter_add(0, sites.long().remainder(size), sw * wp * vals)
    return result


def oracle(x, p, modes, connected):
    vi = torch.stack([scalar_axis(row, x.shape[1], modes, connected, 3) for row in p]).T
    vo = torch.stack([scalar_axis(row, x.shape[1], modes, connected, 2) for row in p]).T
    return (x @ vi * p[:, 0]) @ vo.T


def correctness(args, run):
    from .run import PlanLinear

    reports = []
    size = run.case.size
    for entry in run.plans:
        if getattr(args, "plan_id", None) not in (None, entry.id):
            continue
        recipe = entry.plan.recipe
        p = parameters(size, 20, 83).double()
        p[:, 1] = p.new_tensor(
            [
                0.5,
                0.72,
                0.8,
                0.9,
                1,
                1.44,
                1.6,
                1.8,
                2,
                3.1,
                7,
                12.9,
                16,
                31.9,
                32,
                50,
                0.799,
                0.801,
                1.79,
                1.81,
            ]
        )
        p[:, 2] = 7.5
        p[:, 3] = 13.5
        # Exact FP32 input state is promoted to the independent FP64 oracle.
        p = p.float().double().requires_grad_()
        torch.manual_seed(73)
        x = torch.randn(3, size).double().requires_grad_()
        dy = torch.randn_like(x)
        expected = oracle(x, p, recipe.modes, recipe.connected)
        ex, ep = torch.autograd.grad(expected, (x, p), dy)
        tx = x.detach().float().cuda().requires_grad_()
        model = PlanLinear(
            p.detach().float().cuda(),
            operator(size, recipe.modes, recipe.connected),
            entry.plan,
        )
        actual = model(tx)
        gx, gp = torch.autograd.grad(actual, (tx, model.p), dy.float().cuda())
        errors = {
            n: error(a, e)
            for n, a, e in [
                ("Y", actual.cpu(), expected),
                ("dX", gx.cpu(), ex),
                ("dP", gp.cpu(), ep),
            ]
        }
        for col, n in enumerate(("amplitude", "radius", "mu_out", "mu_in")):
            errors[n] = error(gp[:, col].cpu(), ep[:, col])
        if any(e["relative_l2"] > 3e-4 or e["max_abs"] > 3e-4 for e in errors.values()):
            raise AssertionError(errors)
        # Compare full AdamW trajectories, including moments/counters. Both
        # references start from identical FP32 rows (oracle promotes locally).
        cp = torch.nn.Parameter(p.detach().float()[:6].clone())
        gm = PlanLinear(
            cp.detach().cuda(),
            operator(size, recipe.modes, recipe.connected),
            entry.plan,
        )
        cop = torch.optim.AdamW([cp], lr=1e-4, weight_decay=0, fused=False)
        gop = torch.optim.AdamW(gm.parameters(), lr=1e-4, weight_decay=0, fused=True)
        for _ in range(4):
            cop.zero_grad(set_to_none=True)
            gop.zero_grad(set_to_none=True)
            cy = oracle(x.detach(), cp.double(), recipe.modes, recipe.connected)
            gy = gm(tx.detach())
            cy.square().mean().backward()
            gy.square().mean().backward()
            cop.step()
            gop.step()
        trajectory = {"parameter": error(gm.p.cpu(), cp)}
        for name in ("exp_avg", "exp_avg_sq", "step"):
            trajectory[name] = error(gop.state[gm.p][name].cpu(), cop.state[cp][name])
        if (
            trajectory["parameter"]["max_abs"] > 2e-4
            or trajectory["exp_avg"]["relative_l2"] > 5e-4
            or trajectory["exp_avg_sq"]["relative_l2"] > 1e-3
            or trajectory["step"]["max_abs"] != 0
        ):
            raise AssertionError(trajectory)
        reports.append(
            {"plan": entry.id, "errors": errors, "AdamW_4_steps": trajectory}
        )
    return {
        "status": "PASS",
        "scope": "independent FP64 Chebyshev quadrature Y/dX/all physical gradients, 20 boundary/wide/repeated-image atoms, FP32-encoded state",
        "plans": reports,
    }


def measure_case(args, run):
    from .run import PlanLinear

    case = run.case
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    p = parameters(case.size, case.atoms, case.seed)
    torch.manual_seed(case.seed + 1)
    xx = torch.randn(case.rows, case.size)
    target = torch.randn_like(xx)
    hashes = {n: tensor_hash(v) for n, v in [("p", p), ("x", xx), ("target", target)]}
    dense = args.worker == "dense"
    if dense:
        model = torch.nn.Linear(case.size, case.size, bias=False).cuda()
        # Dense uses its own standard initialization. It is a cost comparator,
        # not asserted to be the same function as the new mathematical fixture.
    else:
        entry = run.entry(args.plan_id)
        recipe = entry.plan.recipe
        model = PlanLinear(
            p.cuda(), operator(case.size, recipe.modes, recipe.connected), entry.plan
        )
    x = xx.cuda().requires_grad_()
    target = target.cuda()
    opt = torch.optim.AdamW(
        model.parameters(), lr=case.optimizer.lr, weight_decay=0, fused=True
    )

    def step():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) - target).square().mean()
        loss.backward()
        opt.step()
        if not dense:
            with torch.no_grad():
                model.p[:, 1].clamp_(0.5, min(50, case.size / 2))
                model.p[:, 2:].remainder_(case.size)
        return loss

    for _ in range(case.warmup):
        step()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(case.rounds):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    state = {
        str(i): {
            n: tensor_hash(v)
            for n, v in opt.state[q].items()
            if isinstance(v, torch.Tensor)
        }
        for i, q in enumerate(model.parameters())
    }
    if not bool(torch.isfinite(loss)):
        raise AssertionError("nonfinite benchmark loss")
    return {
        "status": "PASS",
        "reference": "dense_linear" if dense else args.plan_id,
        "eager": {"median_ms": statistics.median(samples), "samples_ms": samples},
        "graph": None,
        "peak_allocated_training_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_training_bytes": torch.cuda.max_memory_reserved(),
        "total_gpu_process_bytes": None,
        "initial_p_sha256": hashes["p"],
        "initial_inputs": hashes,
        "optimizer": asdict(case.optimizer),
        "optimizer_policy": "common fused AdamW raw physical rows, no decay; radius clamp [.5,50], centers wrap; dense standard AdamW. This is a new fixture, not old Polar update semantics.",
        "scope": "complete eager forward/loss/backward/AdamW including dX and geometry construction; no Graph; IEEE FP32",
        "memory_scope": "resident model/gradients/AdamW plus temporary geometry, response and FP32 moments; setup and independent proof outside measurement",
        "parameter_payload_bytes": sum(
            q.numel() * q.element_size() for q in model.parameters()
        ),
        "optimizer_payload_bytes": sum(
            v.numel() * v.element_size()
            for st in opt.state.values()
            for v in st.values()
            if isinstance(v, torch.Tensor)
        ),
        "final_optimizer_hashes": state,
        "final_loss": float(loss.detach()),
        "dense_initialization": "standard nn.Linear, compute/memory comparator only"
        if dense
        else None,
    }
