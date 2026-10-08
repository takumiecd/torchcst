"""Untimed, full-atom numerical diagnosis; never a performance runner.

Hardware and FP64 trig use identical FP32 arguments, profile arithmetic, global
normalization, launch shape and VJP. Diagnostic tensors are saved as evidence,
not consumed by the runtime algorithm.
"""

import copy
import json
from pathlib import Path

import torch


def metrics(actual, truth, tol=4e-4):
    a, b = actual.detach().double(), truth.detach().double()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("nonfinite numerical diagnostic")
    error = (a - b).abs()
    index = int(error.reshape(-1).argmax())
    coordinate = list(torch.unravel_index(torch.tensor(index), error.shape))
    elementwise = error > tol + tol * b.abs()
    maximum = float(error.max())
    relative = float(error.norm() / b.norm().clamp_min(1e-30))
    return {
        "max": maximum,
        "rel_l2": relative,
        "elementwise_failures": int(elementwise.sum()),
        "index": [int(i) for i in coordinate],
        "actual": float(a.reshape(-1)[index]),
        "truth": float(b.reshape(-1)[index]),
        "pass": maximum <= tol and relative <= tol and not bool(elementwise.any()),
    }


def diagnose(output, *, source):
    import triton

    from torchcst import CSTLinear
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.algorithm import (
        OnchipRecipe,
    )
    from torchcst._backends.cuda.algorithms.linear.torus_profile_product.onchip.executor import (
        onchip_product,
    )
    from torchcst._backends.torch.algorithms.linear.torus_profile_product.algorithm import (
        TorusChunkRecipe,
    )
    from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
        _queries,
        _Scalars,
        chunked_product,
    )
    from torchcst._backends.torch.parameterizations import polar_amp_width as polar

    from .manifest import load_run
    from .torus_onchip_numerics_probe import probe
    from .torus_profile_product import fixture_operator, initialize, oracle_vjp

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": source,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "performance_measured": False,
        "tol": 4e-4,
        "cases": [],
    }
    assert report["gpu"] == "NVIDIA L4"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for n in (1024, 2048):
        case = load_run(
            f"benchmarks/cuda/linear/cases/torus-profile-product-onchip-{n}-sigma3.json",
            "benchmarks/cuda/linear/plans-torus-profile-product-onchip.json",
        ).case
        op = fixture_operator(case)
        layer = CSTLinear(
            chart=op.charts[0], kernel=op.kernel, atoms=initialize(case), device="cuda"
        )
        p = layer.atoms.p
        gen = torch.Generator().manual_seed(case.seed)
        x = torch.randn(case.rows, n, generator=gen).cuda().requires_grad_()
        dy = torch.randn(case.rows, n, generator=gen).cuda()
        truth, tx, tp = oracle_vjp(
            copy.deepcopy(layer.kernel).double(), p, x, dy, layer.chart
        )
        row = {"n": n, "sigma": 3, "atoms": len(p), "results": []}
        report["cases"].append(row)
        payload = {
            "p": p.detach().cpu(),
            "x": x.detach().cpu(),
            "dy": dy.cpu(),
            "truth_y": truth.cpu(),
            "truth_dx": tx.cpu(),
            "truth_dp": tp.cpu(),
        }
        baseline_dp = None
        # Replicate atomically accumulated tensors; dP has no atomics.
        for mode in ("hardware", "fp64"):
            for repeat in range(2):
                y = onchip_product(
                    x, p, layer.kernel, layer.chart, OnchipRecipe(trig=mode)
                )
                gx, gp = torch.autograd.grad(y, (x, p), dy)
                result = {
                    "mode": mode,
                    "repeat": repeat,
                    "y": metrics(y, truth),
                    "dx": metrics(gx, tx),
                    "dp": metrics(gp, tp),
                }
                row["results"].append(result)
                payload[f"{mode}-{repeat}"] = {
                    "y": y.detach().cpu(),
                    "dx": gx.cpu(),
                    "dp": gp.cpu(),
                }
                if mode == "hardware" and repeat == 0:
                    baseline_dp = gp.detach().clone()
                print(json.dumps({"n": n, **result}), flush=True)
                (output / "numerics.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
        # Current passing Torch route is a control, not an alternate oracle.
        y = chunked_product(
            x, p, layer.kernel, layer.chart, TorusChunkRecipe(atom_chunk=1024)
        )
        gx, gp = torch.autograd.grad(y, (x, p), dy)
        result = {
            "mode": "torch-h",
            "y": metrics(y, truth),
            "dx": metrics(gx, tx),
            "dp": metrics(gp, tp),
        }
        row["results"].append(result)
        payload["torch-h"] = {"y": y.detach().cpu(), "dx": gx.cpu(), "dp": gp.cpu()}
        assert all(result[key]["pass"] for key in ("y", "dx", "dp")), result
        # Find worst atoms independently of the changed-trig results.
        error = (baseline_dp.double() - tp).abs()
        indices = error.max(dim=1).values.topk(16).indices
        pp = p.detach()[indices].contiguous()
        scalars = _Scalars(layer.kernel, pp)
        amp, alpha = polar._amplitude_and_alpha(scalars, pp[:, :2])
        sigma, _, _ = polar._sigma_bounds(scalars, amp, alpha, side="input")
        precision = sigma.reciprocal().square().detach()
        major, minor, circle, sites, _ = _queries(layer.chart, pp)
        probes = {}
        for mode in ("hardware", "fp64"):
            centre = pp.new_empty((len(pp), 8))
            u, v = pp.new_empty((len(pp), n)), pp.new_empty((len(pp), n))
            probe[(len(pp),)](
                pp,
                precision,
                scalars.scalar("amplitude_max"),
                major,
                minor,
                circle,
                sites,
                centre,
                u,
                v,
                N=n,
                TRIG=mode,
                T=64,
                num_warps=4,
                enable_fp_fusion=False,
            )
            probes[mode] = {"centre": centre.cpu(), "u": u.cpu(), "v": v.cpu()}
        payload["worst_atoms"] = indices.cpu()
        payload["probes"] = probes
        row["worst_atoms"] = indices.cpu().tolist()
        row["trig_probe_difference"] = {
            key: metrics(probes["hardware"][key], probes["fp64"][key])
            for key in ("centre", "u", "v")
        }
        torch.save(payload, output / f"numerics-{n}.pt")
        (output / "numerics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
