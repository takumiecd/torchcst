"""Compare Torch and research CUDA execution of the existing two-Sphere Polar contract.

The factor-weight route assembles W from the existing factors without the
public materialized route's [atoms, output, input] temporary. All training
measurements use the actual eager CSTOptimizer, including validation and
Sphere retraction/transport. This is not the CUDA-Graph Linear runner schema.
"""

import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from torchcst import (
    BandwidthBounds,
    CSTLinear,
    CSTOptimizer,
    TriweightSpec,
    chart_presets,
    geometry_presets,
    presets,
)

ROUTES = (
    "factored",
    "factor-weight",
    "materialized",
    "blocked-256",
    "blocked-1024",
    "blocked-4096",
    "blocked-16384",
    "blocked-recompute",
    "support-16",
    "support-64",
    "support-256",
    "weight-16",
    "weight-64",
    "weight-256",
    "weight-64-tile64",
)


def fixture(size, sigma, *, batch=32, seed=41, atoms=None):
    """S2 x S2, random uniform sites, unit surface-site density, intrinsic centers."""
    torch.manual_seed(seed)
    radius = math.sqrt(size / (4 * math.pi))
    charts = []
    for _ in range(2):
        sites = torch.randn(size, 3)
        sites = sites / sites.norm(dim=-1, keepdim=True) * radius
        charts.append(
            chart_presets.points(
                sites.tolist(),
                geometry=geometry_presets.sphere(
                    2, radius=radius, representation="intrinsic"
                ),
            )
        )
    bounds = BandwidthBounds(minimum=1.0, birth=1.0, maximum=16.0, upper_floor=1.0)
    kernel = presets.polar_activity(
        amplitude_max=1.0,
        input_bounds=bounds,
        w_c=1e6,
        profile=presets.profile(TriweightSpec()),
        dormant_expansion_rate=0.02,
    )
    model = CSTLinear(
        *charts,
        atoms=atoms or max(1, int(size * size * 0.05)),
        kernel=kernel,
        backend="factored",
    )
    # Same amplitude distribution and width/activity law as the large Polar
    # fixtures. Preserve the existing uniform/balanced Sphere center sampling.
    gen = torch.Generator().manual_seed(seed + 177)
    direction = torch.rand(model.atom_count, generator=gen) * 0.4 - 0.2
    alpha = math.log(sigma) / math.log(16)
    radius_p = math.sqrt(1 + 3 * alpha)
    with torch.no_grad():
        model.atoms.p[:, :2] = (
            torch.stack((direction, (1 - direction.square()).sqrt()), -1) * radius_p
        )
    x = torch.randn(batch, size, generator=gen)
    target = torch.randn(batch, size, generator=gen)
    dy = torch.randn(batch, size, generator=gen)
    return model, x, target, dy


def forward(model, x, route):
    if route.startswith(("blocked-", "support-", "weight-")):
        from benchmarks.cuda.linear.manifest import REGISTRY
        from torchcst._backends.cuda.algorithms.linear.sphere_polar.algorithm import (
            SphereRecipe,
        )
        from torchcst._backends.dispatch import Dispatcher
        from torchcst._backends.schema import ExecutionPlan
        from torchcst.operators.execution import LinearBinding, LinearInputs

        chunk = (
            4096
            if route == "blocked-recompute" or route.startswith(("support-", "weight-"))
            else int(route.split("-")[1])
        )
        plan = ExecutionPlan(
            "research_cuda_sphere_polar_blocked",
            "v1",
            SphereRecipe(chunk, route != "blocked-recompute"),
        )
        if route.startswith(("support-", "weight-")):
            from torchcst._backends.cuda.algorithms.linear.sphere_polar.support_algorithm import (
                SphereSupportRecipe,
            )

            plan = ExecutionPlan(
                (
                    "research_cuda_sphere_polar_weight"
                    if route.startswith("weight-")
                    else "research_cuda_sphere_polar_support"
                ),
                "v1",
                SphereSupportRecipe(int(route.split("-")[1])),
            )
        if route.startswith("weight-"):
            from torchcst._backends.cuda.algorithms.linear.sphere_polar.weight_algorithm import (
                SphereWeightRecipe,
            )

            tile = int(route.split("-")[-1][4:]) if "-tile" in route else 32
            plan = ExecutionPlan(
                "research_cuda_sphere_polar_weight",
                "v2",
                SphereWeightRecipe(int(route.split("-")[1]), tile),
            )
        if not hasattr(model, "_sphere_binding"):
            model._sphere_binding = LinearBinding(model.operator, model.atoms.p)
        return Dispatcher(registry=REGISTRY).run(
            model._sphere_binding, LinearInputs(x), plan=plan
        )
    if route in ("factored", "planned"):
        return model(x)
    if route == "factor-weight":
        vi, uo = model.operator.factors()
        return torch.nn.functional.linear(x, uo @ vi.T)
    if route == "materialized":
        return model.operator.apply(x, algorithm="materialized")
    raise ValueError(route)


def oracle_factors(model, p, *, scale_amplitude=True):
    """Independent embedded-distance and Polar algebra; no backend math calls."""
    spec = model.kernel.spec
    pars = spec.parameterization
    q = p[:, :2].square().sum(-1)
    amp = pars.amplitude_max * p[:, 0] / q.clamp_min(torch.finfo(p.dtype).tiny).sqrt()
    alpha = ((q - 1) / 3).clamp(0, 1)
    z = (amp / pars.w_c).square()
    values = []
    for center, chart, bounds in zip(
        (p[:, 2:4], p[:, 4:6]),
        model.cst_charts(),
        (pars.input_bounds, pars.output_bounds),
    ):
        radius = chart.geometry.radius.double()
        theta = center.norm(dim=-1, keepdim=True) / radius
        embedded = torch.cat(
            (radius * theta.cos(), torch.sinc(theta / torch.pi) * center), -1
        )
        sites = chart.coordinates.double()
        sites = sites * (radius / sites.norm(dim=-1, keepdim=True))
        lo = bounds.minimum + (bounds.birth - bounds.minimum) / (
            1 + pars.lower_kappa * z
        )
        hi = bounds.minimum + (bounds.maximum - bounds.minimum) * pars.kappa / (
            pars.kappa + z.pow(pars.upper_decay_power)
        )
        hi = hi.clamp_min(bounds.upper_floor)
        hi = torch.maximum(hi, lo)
        sigma = (
            torch.exp((1 - alpha) * lo.log() + alpha * hi.log())
            .clamp(min=lo, max=hi)
            .detach()
        )
        dist = (sites[:, None] - embedded[None]).square().sum(-1)
        raw = (1 - dist * sigma.reciprocal().square()[None]).clamp_min(0).pow(3)
        # Existing separable Sphere contract floors each profile separately.
        floor = spec.profiles[len(values)].normalization.floor
        values.append(raw / raw.norm(dim=0).clamp_min(floor)[None])
    # Untimed support diagnostics retain positive support at zero amplitude;
    # sigma still follows the actual amplitude/activity law.
    return values[0], values[1] * amp[None] if scale_amplitude else values[1]


def oracle_vjp(model, x, dy, *, chunk=256):
    tx = x.detach().double().requires_grad_()
    y = tx.new_zeros((len(x), model.out_features))
    dx = torch.zeros_like(tx)
    grads = []
    for start in range(0, model.atom_count, chunk):
        p = model.atoms.p[start : start + chunk].detach().double().requires_grad_()
        vi, uo = oracle_factors(model, p)
        part = (tx @ vi) @ uo.T
        gx, gp = torch.autograd.grad(part, (tx, p), dy.double())
        y += part.detach()
        dx += gx
        grads.append(gp)
    return y, dx, torch.cat(grads)


def error(actual, expected):
    delta = actual.detach().double() - expected.detach().double()
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            delta.norm() / expected.detach().double().norm().clamp_min(1e-30)
        ),
    }


def correctness(model, x, dy, route):
    ox, dx, dp = oracle_vjp(model, x, dy)
    tx = x.detach().clone().requires_grad_()
    y = forward(model, tx, route)
    gx, gp = torch.autograd.grad(y, (tx, model.atoms.p), dy)
    errors = {
        k: error(a, b)
        for k, a, b in zip(("Y", "dX", "all_dP"), (y, gx, gp), (ox, dx, dp))
    }
    passed = all(
        v["max_abs"] <= 4e-4 and v["relative_l2"] <= 4e-4 for v in errors.values()
    )
    if not passed:
        raise AssertionError(errors)
    return {"status": "PASS", "scope": "full_sites_all_atoms_fp64", "errors": errors}


def optimizer(model, *, lr=1e-4, weight_decay=0.01):
    return CSTOptimizer(
        torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, fused=True
        ),
        model=model,
    )


def step(model, opt, x, target, route):
    opt.zero_grad(set_to_none=True)
    x.grad = None
    loss = (forward(model, x, route) - target).square().mean()
    loss.backward()
    opt.step()


def trajectory_gate(size=32, sigma=3.0, steps=20, *, device="cpu"):
    base, x, target, _ = fixture(size, sigma, batch=4)
    a, b = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device)
    # CPU fused AdamW is supported, but a normal optimizer avoids backend-specific
    # optimizer differences in this isolated algebra/layout gate.
    oa = CSTOptimizer(
        torch.optim.AdamW(a.parameters(), lr=1e-4, foreach=False), model=a
    )
    ob = CSTOptimizer(
        torch.optim.AdamW(b.parameters(), lr=1e-4, foreach=False), model=b
    )
    xa, xb = x.to(device).requires_grad_(), x.to(device).requires_grad_()
    target = target.to(device)
    largest = {"parameters": 0.0, "exp_avg": 0.0, "exp_avg_sq": 0.0, "dX": 0.0}
    for _ in range(steps):
        step(a, oa, xa, target, "factored")
        step(b, ob, xb, target, "factor-weight")
        for name, aa, bb in (
            ("parameters", a.atoms.p, b.atoms.p),
            ("dX", xa.grad, xb.grad),
            ("exp_avg", oa.state[a.atoms.p]["exp_avg"], ob.state[b.atoms.p]["exp_avg"]),
            (
                "exp_avg_sq",
                oa.state[a.atoms.p]["exp_avg_sq"],
                ob.state[b.atoms.p]["exp_avg_sq"],
            ),
        ):
            value = error(aa, bb)["max_abs"]
            largest[name] = max(largest[name], value)
            if value > (2e-6 if name == "parameters" else 4e-4):
                raise AssertionError((name, value))
        assert torch.equal(oa.state[a.atoms.p]["step"], ob.state[b.atoms.p]["step"])
    return {"status": "PASS", "steps": steps, "max_abs": largest}


def tensor_hash(tensor):
    # Preserve the historical raw-byte SHA without requiring NumPy (which is
    # not a core/dev dependency). Reshape handles scalars; the byte view limits
    # hashing to tensor contents even when a contiguous slice has extra storage.
    cpu = tensor.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(bytes(cpu.view(torch.uint8).tolist())).hexdigest()


def run_worker(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    base, xx, target, dy = fixture(
        args.size,
        args.sigma,
        batch=getattr(args, "batch", 32),
        seed=getattr(args, "seed", 41),
        atoms=getattr(args, "atoms", None),
    )
    inputs = {
        "p": tensor_hash(base.atoms.p),
        "x": tensor_hash(xx),
        "target": tensor_hash(target),
        "sites": [tensor_hash(c.coordinates) for c in base.cst_charts()],
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = base.to("cuda")
    if hasattr(args, "plan"):
        from benchmarks.cuda.linear.manifest import REGISTRY
        from torchcst._backends.dispatch import FixedSelector

        model.selector = FixedSelector(args.plan, registry=REGISTRY)
    x, target, dy = xx.cuda().requires_grad_(), target.cuda(), dy.cuda()
    args.stage = "initial_full_correctness"
    proof = correctness(model, x, dy, args.route)
    opt = optimizer(
        model,
        lr=getattr(args, "lr", 1e-4),
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    # Oracle scratch is excluded; complete training-state allocation remains.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    args.stage = "training"
    for _ in range(getattr(args, "warmup", 3)):
        step(model, opt, x, target, args.route)
    torch.cuda.synchronize()
    samples = []
    for _ in range(getattr(args, "rounds", 21)):
        torch.cuda.synchronize()
        t = time.perf_counter()
        step(model, opt, x, target, args.route)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1000)
    peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    args.stage = "updated_full_correctness"
    post = correctness(model, x, dy, args.route)
    from torchcst._backends.torch.parameterizations import polar_amp_width

    widths = polar_amp_width.bandwidth_sigmas(
        model.kernel, *model.cst_charts(), model.atoms.p
    )
    widths_summary = [
        {"min": float(w.min()), "max": float(w.max()), "mean": float(w.mean())}
        for w in widths
    ]
    phases = {"forward_loss": [], "backward": [], "optimizer": []}
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = (forward(model, x, args.route) - target).square().mean()
        torch.cuda.synchronize()
        phases["forward_loss"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        phases["backward"].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        phases["optimizer"].append((time.perf_counter() - t) * 1000)
    opt.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    args.stage = "inference"
    inference = []
    with torch.no_grad():
        for _ in range(getattr(args, "warmup", 3)):
            forward(model, x, args.route)
        for _ in range(getattr(args, "rounds", 21)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = forward(model, x, args.route)
            torch.cuda.synchronize()
            inference.append((time.perf_counter() - t) * 1000)
            del y
    return {
        "status": "PASS",
        "size": args.size,
        "sigma_initial": args.sigma,
        "route": args.route,
        "atoms": model.atom_count,
        "batch": len(x),
        "geometry": "two S2 charts; intrinsic centers; uniform random surface sites",
        "radius": math.sqrt(args.size / (4 * math.pi)),
        "normalization": "per-side chart-site discrete L2 with individual 1e-6 floors",
        "protocol": "eager public CSTOptimizer; FP32 IEEE; AdamW; dX included",
        "input_hashes": inputs,
        "correctness_initial": proof,
        "correctness_updated": post,
        "training_ms": {"median": statistics.median(samples), "samples": samples},
        "training_peak_bytes": peak,
        "inference_ms": {"median": statistics.median(inference), "samples": inference},
        "inference_peak_bytes": {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        },
        "inference_memory_scope": "includes resident trained model and optimizer state",
        "updated_widths": widths_summary,
        "phase_ms_separate_instrumented_steps": {
            name: {"median": statistics.median(values), "samples": values}
            for name, values in phases.items()
        },
        "final_p_hash": tensor_hash(model.atoms.p),
        "materialized_atom_tensor_lower_bound_bytes": model.atom_count
        * args.size**2
        * 4,
    }


def run_dense(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    _, xx, target, _ = fixture(
        args.size,
        args.sigma,
        batch=getattr(args, "batch", 32),
        seed=getattr(args, "seed", 41),
        atoms=getattr(args, "atoms", None),
    )
    torch.manual_seed(getattr(args, "seed", 41))
    model = torch.nn.Linear(args.size, args.size, bias=False).cuda()
    x, target = xx.cuda().requires_grad_(), target.cuda()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(args, "lr", 1e-4),
        weight_decay=getattr(args, "weight_decay", 0.01),
        fused=True,
    )

    def train():
        opt.zero_grad(set_to_none=True)
        x.grad = None
        loss = (model(x) - target).square().mean()
        loss.backward()
        opt.step()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(getattr(args, "warmup", 3)):
        train()
    samples = []
    for _ in range(getattr(args, "rounds", 21)):
        torch.cuda.synchronize()
        t = time.perf_counter()
        train()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1000)
    return {
        "status": "PASS",
        "size": args.size,
        "sigma_initial": args.sigma,
        "route": "dense",
        "input_hashes": {"x": tensor_hash(x), "target": tensor_hash(target)},
        "protocol": "eager FP32 IEEE AdamW; different model",
        "training_ms": {"median": statistics.median(samples), "samples": samples},
        "training_peak_bytes": {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--route", choices=(*ROUTES, "dense"), default="factored")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-gate", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.trajectory_gate:
            result = trajectory_gate(device="cuda")
        elif args.route == "dense":
            result = run_dense(args)
        else:
            result = run_worker(args)
    except torch.OutOfMemoryError as exc:
        result = {
            "status": "OOM",
            "size": args.size,
            "sigma_initial": args.sigma,
            "route": args.route,
            "error": str(exc),
            "stage": getattr(args, "stage", "setup"),
            "incomplete_peak_bytes": {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            },
        }
    except Exception as exc:
        result = {
            "status": "FAIL",
            "size": args.size,
            "sigma_initial": args.sigma,
            "route": args.route,
            "error": repr(exc),
            "stage": getattr(args, "stage", "setup"),
        }
        args.output.write_text(json.dumps(result, indent=2))
        raise
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "status",
                    "size",
                    "sigma_initial",
                    "route",
                    "training_ms",
                    "training_peak_bytes",
                )
            }
        ),
        flush=True,
    )


def measure_case(args, run):
    """Use the existing Linear runner with an explicit eager Sphere policy.

    Public Sphere CSTOptimizer contains host validation and rejects capture.
    Linear Graph correctness is tested separately; this result claims no
    captured full optimizer step and is not the existing submission adapter.
    """
    from dataclasses import asdict
    from types import SimpleNamespace

    case = run.case
    config = SimpleNamespace(
        size=case.size,
        sigma=float(case.profile[3:]),
        batch=case.rows,
        seed=case.seed,
        atoms=case.atoms,
        warmup=case.warmup,
        rounds=case.rounds,
        lr=case.optimizer.lr,
        weight_decay=case.optimizer.weight_decay,
        route="planned",
    )
    if args.worker == "dense":
        result = run_dense(config)
    else:
        config.plan = run.entry(args.plan_id).plan
        result = run_worker(config)
    hashes = result["input_hashes"]
    return {
        "status": result["status"],
        "reference": "dense_linear" if args.worker == "dense" else args.plan_id,
        "eager": {
            "median_ms": result["training_ms"]["median"],
            "samples_ms": result["training_ms"]["samples"],
        },
        "graph": None,
        "initial_p_sha256": hashes.get("p"),
        "initial_inputs": {"x_sha256": hashes["x"], "target_sha256": hashes["target"]},
        "peak_allocated_training_bytes": result["training_peak_bytes"]["allocated"],
        "peak_reserved_training_bytes": result["training_peak_bytes"]["reserved"],
        "total_gpu_process_bytes": None,
        "optimizer": asdict(case.optimizer),
        "optimizer_policy": "eager public CSTOptimizer; intrinsic S2 retraction and Polar activity",
        "scope": "complete eager step with full-site all-atom FP64 checks before and after training",
        "memory_scope": "eager model, gradients and AdamW state; no Graph capture; GPU process usage unmeasured",
        "sphere_diagnostics": result,
    }


if __name__ == "__main__":
    main()
