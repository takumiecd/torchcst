"""Numerical admission probe for one FP32 continuous-body candidate.

No timings are collected here. A completed probe can report a failed candidate.
"""

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.cuda.linear.radius_cell_mass import (
    _reference_model,
    chunked_reference,
    fixture,
)
from torchcst import CSTAdamW, FixedSelector, make_algorithm_registry
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
    hierarchy_from_operator,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.executor import (
    _extension,
)
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_OWNER_WINDOW_RECOMPUTE,
)
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)


def errors(actual, reference):
    a, r = actual.detach().cpu().double(), reference.detach().cpu().double()
    d = (a - r).abs()
    scale = 3e-5 + 3e-4 * r.abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    if not d.numel():
        return {"pass": finite, "elements": 0}
    index = int((d / scale).reshape(-1).argmax())
    return {
        "pass": finite and bool((d <= scale).all()),
        "finite": finite,
        "elements": d.numel(),
        "bad_elements": int((d > scale).sum()),
        "max_abs": float(d.max()),
        "rms": float(d.square().mean().sqrt()),
        "p99_abs": float(torch.quantile(d.reshape(-1), 0.99)),
        "relative_l2": float(d.norm() / r.norm().clamp_min(1e-300)),
        "worst_scaled_error": float((d / scale).max()),
        "worst_flat_index": index,
        "actual_at_worst": float(a.reshape(-1)[index]),
        "reference_at_worst": float(r.reshape(-1)[index]),
    }


def quadrature(model, x, p):
    """Independent Gauss-Legendre integration, respecting the public FP32 tiny.

    Radius and discrete support are detached; center and amplitude remain live.
    Unlike the Fourier GPU code, this integrates a Chebyshev polynomial directly.
    """
    import numpy as np

    safe = p[:, :2].square().sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    amp = p[:, 0] / safe.sqrt()
    alpha = ((safe - 1) / 3).clamp(0, 1)
    w_c = float(model.kernel.scalar("w_c"))
    kappa = float(model.kernel.scalar("kappa"))
    lower_kappa = float(model.kernel.scalar("lower_kappa"))
    power = float(model.kernel.scalar("upper_decay_power"))
    z = (amp / w_c).square()
    lower = 0.5 + 3.5 / (1 + lower_kappa * z)
    upper = (0.5 + 49.5 * kappa / (kappa + z.pow(power))).clamp_min(1)
    upper = torch.maximum(lower, upper)
    radius = ((1 - alpha) * lower.log() + alpha * upper.log()).exp().detach()
    nodes, weights = np.polynomial.legendre.leggauss(48)
    nodes, weights = p.new_tensor(nodes), p.new_tensor(weights)
    factors = []
    for column in (3, 2):
        rows = []
        for r, mu in zip(radius, p[:, column]):
            tile = 1
            while float(r) > tile:
                tile *= 2
            count = math.ceil(2 * float(r) / tile) + 1
            period = count * tile
            c = mu.remainder(1024)
            first = (
                math.floor((float(c.detach()) + 0.5) / tile - (count - 1) / 2) * tile
            )
            sites = torch.arange(first, first + period, dtype=p.dtype)

            def integrate(left, right, r=r, period=period):
                u = (right - left)[..., None] / 2 * nodes + (right + left)[
                    ..., None
                ] / 2
                beta = 1 / torch.cos(torch.pi * r / period)
                v = beta * torch.cos(torch.pi * u / period)
                a, b = torch.ones_like(v), v
                for _ in range(2, 9):
                    a, b = b, 2 * v * b - a
                value = b / torch.cosh(8 * torch.acosh(beta))
                return (right - left) / 2 * (value.square() * weights).sum(-1)

            left = sites - 0.5 - c
            mass = integrate(left, left + 1) / integrate(
                p.new_tensor(-period / 2), p.new_tensor(period / 2)
            )
            rows.append(
                p.new_zeros(1024).scatter_add(0, sites.long().remainder(1024), mass)
            )
        factors.append(torch.stack(rows).T)
    return (x @ factors[0]) @ (factors[1] * amp).T


def vjp_reference(model, x, dy, independent=False):
    m = copy.deepcopy(model).cpu().double()
    m.selector = None
    m.backend = "factored"
    rx = x.double().requires_grad_()
    if not independent:
        return chunked_reference(m, rx, dy.double())
    y = quadrature(m, rx, m.atoms.p)
    return (y.detach(), *torch.autograd.grad(y, (rx, m.atoms.p), dy.double()))


def adam_direction(g, m=None, v=None, step=0, eps=1e-8):
    # Match FP32 moment buffers and bias corrections, independent of CST kernels.
    if m is None:
        m, v = torch.zeros_like(g), torch.zeros_like(g)
    m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g.square()
    return (m / (1 - 0.9 ** (step + 1))) / (
        (v / (1 - 0.999 ** (step + 1))).sqrt() + eps
    )


def optimizer_check(model, actual, reference):
    """Cold Adam sensitivity plus real CSTAdamW post-Polar state comparison."""
    g, r = actual.cpu().float(), reference.cpu().float()
    u, ur = adam_direction(g), adam_direction(r)
    du = (u - ur).abs()
    rms = du.double().square().mean(0).sqrt()
    eps_band = (r.abs() >= 1e-9) & (r.abs() <= 1e-7)
    result = {
        "pass": bool(torch.isfinite(du).all())
        and float(du.max()) <= 0.01
        and bool((rms <= 0.001).all()),
        "max_direction_error": float(du.max()),
        "direction_rms_by_column": rms.tolist(),
        "eps_band_count": int(eps_band.sum()),
        "eps_band_sign_changes": int(((g.sign() != r.sign()) & eps_band).sum()),
        "zeroed_nonzero_gradients": int(((g == 0) & (r != 0)).sum()),
    }
    from benchmarks.training.cell_mass_mnist import radius_optimizer_contract

    contract = radius_optimizer_contract()
    result["optimizer_contract"] = contract
    states = []
    for grad in (g, r, r):
        m = _reference_model(model, model.atoms.p).float().cuda()
        opt = CSTAdamW(
            m.parameters(),
            model=m,
            lr=contract["lr"],
            weight_decay=contract["atom_raw_weight_decay"],
            amplitude_weight_decay=contract["amplitude_weight_decay"],
            fused=contract["fused"],
            capturable=contract["capturable"],
            betas=contract["betas"],
            eps=contract["eps"],
            amsgrad=contract["amsgrad"],
            maximize=contract["maximize"],
            check_finite=contract["check_finite"],
            update_selector=FixedSelector(
                RADIUS_FLAT_TORUS_FUSED, registry=make_algorithm_registry()
            ),
        )
        m.atoms.p.grad = grad.cuda()
        opt.step()
        states.append(
            (m.atoms.p.detach().cpu().clone(), copy.deepcopy(opt.state_dict()))
        )
    a, b, control = [s[0] for s in states]
    ulp = (torch.nextafter(b, torch.full_like(b, float("inf"))) - b).abs()
    tolerance = 1e-6 + 2 * ulp
    distance = (a.double() - b.double()).abs()
    distance[:, 2:] = (
        (a[:, 2:].double() - b[:, 2:].double() + 512).remainder(1024) - 512
    ).abs()
    finite_control = torch.isfinite(b).all(1)
    result["baseline_post_p_nonfinite_rows"] = (
        (~finite_control).nonzero().flatten().tolist()
    )
    result["candidate_post_p_nonfinite_rows"] = (
        (~torch.isfinite(a).all(1)).nonzero().flatten().tolist()
    )
    result["post_p_compared_rows"] = int(finite_control.sum())
    result["post_p_max_abs_by_column"] = distance[finite_control].max(0).values.tolist()
    result["post_p_gate"] = bool(
        (distance[finite_control] <= tolerance[finite_control]).all()
    )
    result["full_fixture_admissible"] = bool(finite_control.all())
    result["same_gradient_parameter_bytes"] = bool(
        torch.equal(b.view(torch.uint8), control.view(torch.uint8))
    )
    # Full state including moments/step, not just the updated parameters.
    from benchmarks.training.full_step_graph import exact, tensors

    result["same_gradient_optimizer_state"] = exact(
        dict(tensors(states[1][1])), dict(tensors(states[2][1]))
    )
    result["pass"] &= (
        result["post_p_gate"]
        and result["same_gradient_parameter_bytes"]
        and result["same_gradient_optimizer_state"]["all_byte_equal"]
    )
    return result


def diagnostics(ext, model, x, dy, configs, out):
    data = [
        v.detach().cpu()
        for v in ext.research_owner_window_fp32_diagnose(
            x.cuda(), dy.cuda(), model.atoms.p.detach().cuda(), *configs
        )
    ]
    a, r, shared, meta, wx, wy, dx, dyw = data
    batch = x.size(0)
    fields = {
        "c": (0, 9),
        "q_input": (9, 26),
        "dq_input": (26, 43),
        "q_output": (43, 60),
        "dq_output": (60, 77),
        "H": (77, 77 + batch),
        "Hmu": (77 + batch, 77 + 2 * batch),
        "G": (77 + 2 * batch, 77 + 3 * batch),
        "Gmu": (77 + 3 * batch, 77 + 4 * batch),
        "pre_pullback": (77 + 4 * batch, 80 + 4 * batch),
    }
    result = {
        name: errors(a[:, lo:hi], r[:, lo:hi]) for name, (lo, hi) in fields.items()
    }
    result["fp32_with_fp64_windows_rounded_once"] = {
        name: errors(shared[:, lo:hi], r[:, lo:hi])
        for name, (lo, hi) in fields.items()
        if lo >= 77
    }
    # Explicitly select initialized active owners; never compare torch::empty padding.
    plan = hierarchy_from_operator(model.declaration())
    for label, axis, w, refw, bankcol, ownercol in [
        ("input", plan.input, wx, dx, 0, 1),
        ("output", plan.output, wy, dyw, 2, 3),
    ]:
        offsets, total = [], 0
        for bank in axis.banks:
            offsets.append(total)
            total += bank.tile_count * min(bank.window_tiles * bank.tile_sites, 17)
        selected = []
        for bi, owner in set(map(tuple, meta[:, [bankcol, ownercol]].tolist())):
            bank = axis.banks[bi]
            rank = min(bank.window_tiles * bank.tile_sites, 17)
            first = offsets[bi] + (owner - bank.node_offset) * rank
            selected.extend(range(first, first + rank))
        result[label + "_active_window"] = errors(w[:, selected], refw[:, selected])
    # Save only initialized tables/meta; windows contain inactive uninitialized storage.
    torch.save(
        {
            "candidate": a,
            "reference_gpu_fp64": r,
            "shared_windows": shared,
            "classification": meta,
            "p": model.atoms.p.detach().cpu(),
            "x": x,
            "dy": dy,
        },
        out,
    )
    return result


def run_case(ext, name, model, x, dy, out, *, independent=True):
    print("CASE " + name, flush=True)
    plan = hierarchy_from_operator(model.declaration())
    configs = (plan.input.native_config(), plan.output.native_config())
    expected = vjp_reference(model, x, dy, independent)
    actual = []
    for single in (False, True):
        xx = x.cuda().requires_grad_()
        p = model.atoms.p.detach().cuda().requires_grad_()
        if single:
            y = ext.research_owner_window_fp32(xx, p, *configs, False)
        else:
            y = ext.checked_autograd_linear_owner_window(xx, p, *configs, 2)
        actual.append(
            (
                y.detach().cpu(),
                *[v.cpu() for v in torch.autograd.grad(y, (xx, p), dy.cuda())],
            )
        )
    # Preserve expensive all-atom values before any optimizer/later-stage failure.
    torch.save(
        {
            "expected": expected,
            "baseline": actual[0],
            "candidate": actual[1],
            "p": model.atoms.p.detach().cpu(),
            "x": x,
            "dy": dy,
        },
        out / (name + "-vjp.pt"),
    )
    record = {
        "name": name,
        "atoms": len(model.atoms.p),
        "reference": "quadrature48"
        if independent
        else "chunked_public_CPU_FP64_Fourier",
    }
    for label, values in zip(("baseline_fp64", "candidate_fp32"), actual):
        record[label] = {
            key: errors(a, r) for key, a, r in zip(("Y", "dX", "dP"), values, expected)
        }
        record[label]["dP_by_column"] = {
            key: errors(values[2][:, i], expected[2][:, i])
            for i, key in enumerate(("s", "t", "output_center", "input_center"))
        }
    record["optimizer"] = optimizer_check(model, actual[1][2], expected[2])
    record["pass"] = (
        all(record["candidate_fp32"][key]["pass"] for key in ("Y", "dX", "dP"))
        and record["optimizer"]["pass"]
        and record["optimizer"]["full_fixture_admissible"]
    )
    record["baseline_vjp_pass"] = all(
        record["baseline_fp64"][key]["pass"] for key in ("Y", "dX", "dP")
    )
    record["stages"] = diagnostics(
        ext, model, x, dy, configs, out / (name + "-stages.pt")
    )
    torch.save(
        {
            "expected": expected,
            "baseline": actual[0],
            "candidate": actual[1],
            "p": model.atoms.p.detach().cpu(),
            "x": x,
            "dy": dy,
        },
        out / (name + "-vjp.pt"),
    )
    return record


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ext = _extension()
    rows = [
        [1, 0, 0, 1024],
        [0.3, 0.9539392014, -1024, -0.0001],
        [0.15, 0.9886859967, 1023.99, 1024.01],
        [0.15, 1.33, 4096, -4096],
        [0.15, 1.6, 7.3, 8.2],
        [0, 2, 32, 64],
        [0.3, 1.9773719933, 128, 256],
        [1, 1e-4, 4, 8],
        [1e20, 1e20, 64, 128],
        [1e-8, 1, 512, 768],
    ]
    model, x, _target, dy = fixture(
        SimpleNamespace(size=1024, atoms=len(rows), rows=4, seed=41)
    )
    with torch.no_grad():
        model.atoms.p.copy_(torch.tensor(rows))
    cases = []

    def add(name, m, xx, gg, independent=True):
        cases.append(
            run_case(ext, name, m, xx, gg, args.output, independent=independent)
        )
        (args.output / "results.json").write_text(
            json.dumps(
                {
                    "cases": cases,
                    "all_numerical_gates_pass": all(c["pass"] for c in cases),
                },
                indent=2,
            )
            + "\n"
        )

    for name, xx, gg in [
        ("random", x, dy),
        ("constant", torch.ones_like(x), torch.ones_like(dy)),
        ("alternating", (torch.arange(1024) % 2 * 2 - 1).float().repeat(4, 1), dy),
        (
            "onehot",
            torch.nn.functional.one_hot(torch.tensor([0, 1, 511, 1023]), 1024).float(),
            dy,
        ),
        ("eps_band", x, dy * 1e-7),
        ("zero_signal", torch.zeros_like(x), dy),
    ]:
        add(name, model, xx, gg)
    concentrated, *_ = fixture(SimpleNamespace(size=1024, atoms=129, rows=4, seed=41))
    with torch.no_grad():
        concentrated.atoms.p[:, 0] = 0.15
        concentrated.atoms.p[:, 1] = (1 - 0.15**2) ** 0.5
        concentrated.atoms.p[:, 2:] = 4
    add("concentrated129", concentrated, x, dy)
    boundary = copy.deepcopy(model)
    with torch.no_grad():
        centers = torch.tensor(
            [0.0, 0.5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, 511.5, 1024.0]
        )
        boundary.atoms.p[:, 2] = torch.nextafter(
            centers, torch.full_like(centers, float("inf"))
        )
        boundary.atoms.p[:, 3] = torch.nextafter(
            centers, torch.full_like(centers, -float("inf"))
        )
    add("adjacent_centers", boundary, x, dy)
    for name, polar in [("zero_norm", [0.0, 0.0]), ("tiny_norm", [1e-20, 1e-20])]:
        special, *_ = fixture(SimpleNamespace(size=1024, atoms=1, rows=4, seed=41))
        with torch.no_grad():
            special.atoms.p.copy_(torch.tensor([[*polar, 32.0, 64.0]]))
        add(name, special, x, dy * 1e-10)
    radii = copy.deepcopy(model)
    target = torch.tensor([4.0, 8.0, 16.0, 32.0, 50.0], dtype=torch.float64)
    t = (
        (1 + 3 * (target.log() - math.log(4)) / (math.log(50) - math.log(4)))
        .clamp(1, 4)
        .sqrt()
        .float()
    )
    with torch.no_grad():
        radii.atoms.p[:, 0] = 0
        radii.atoms.p[:, 1] = torch.stack(
            [
                torch.nextafter(t, torch.zeros_like(t)),
                torch.nextafter(t, torch.full_like(t, float("inf"))),
            ],
            1,
        ).flatten()
    add("radius_boundary_neighbors", radii, x, dy)
    if args.inputs:
        from benchmarks.training.cell_mass_mnist import make_model

        init = torch.load(
            args.inputs / "initialization.pt", map_location="cpu", weights_only=True
        )
        for state in ("initial", "final"):
            net = make_model(init, "cst", execution_plan=GLOBAL_OWNER_WINDOW_RECOMPUTE)
            if state == "final":
                net.load_state_dict(
                    torch.load(
                        args.inputs / "baseline_final.pt",
                        map_location="cpu",
                        weights_only=True,
                    )["model"]
                )
            full = net[2]
            full.selector = None
            full.backend = "factored"
            add("full52428_" + state, full, x, dy, False)
    result = {
        "all_numerical_gates_pass": all(c["pass"] for c in cases),
        "all_baseline_vjp_gates_pass": all(c["baseline_vjp_pass"] for c in cases),
        "cases": cases,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "math_threads": torch.get_num_threads(),
        },
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


def recheck_optimizer_saved(args):
    """Re-evaluate saved gradients with the exact frozen training coefficients.

    This does not run or compile the radius extension and does not alter the
    candidate gradients. The normal subset separates existing extreme-norm
    optimizer nonfinites from sensitivity caused by the precision change.
    """
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    for path in sorted(args.inputs.glob("*-vjp.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        p = saved["p"]
        normal = p[:, :2].double().square().sum(1).le(4.000001)
        for label, mask in [
            ("all_rows", torch.ones(len(p), dtype=torch.bool)),
            ("normal_rows", normal),
        ]:
            m, *_ = fixture(
                SimpleNamespace(size=1024, atoms=int(mask.sum()), rows=4, seed=41)
            )
            with torch.no_grad():
                m.atoms.p.copy_(p[mask])
            result = optimizer_check(
                m, saved["candidate"][2][mask], saved["expected"][2][mask]
            )
            rows.append(
                {
                    "case": path.stem,
                    "subset": label,
                    "row_indices": mask.nonzero().flatten().tolist(),
                    **result,
                }
            )
    (args.output / "optimizer-recheck.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "optimizer_rechecks": len(rows),
                "any_failed_normal_gate": any(
                    not r["pass"] for r in rows if r["subset"] == "normal_rows"
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--optimizer-only", action="store_true")
    args = parser.parse_args()
    if args.optimizer_only:
        if args.inputs is None:
            parser.error("--optimizer-only requires --inputs")
        recheck_optimizer_saved(args)
    else:
        run(args)
