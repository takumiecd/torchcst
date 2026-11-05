"""Calibrate FP32 numerical gates against ordinary dense Linear on the same L4.

This compares each representation to its own CPU FP64 oracle. Atom coordinates
and dense weight coordinates differ: max errors cannot rank learning quality.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch

from benchmarks.training.cell_mass_mnist import radius_optimizer_contract
from benchmarks.training.owner_window_fp32 import adam_direction, errors


def case(name, w, x, dy):
    refw, rx, rd = w.double(), x.double(), dy.double()
    reference = (rx @ refw.T, rd @ refw, rd.T @ rx)
    xx, ww = x.cuda().requires_grad_(), w.cuda().requires_grad_()
    y = torch.nn.functional.linear(xx, ww)
    dx, dw = torch.autograd.grad(y, (xx, ww), dy.cuda())
    actual = (y.detach().cpu(), dx.cpu(), dw.cpu())
    row = {
        "case": name,
        "VJP": {
            k: errors(a, r) for k, a, r in zip(("Y", "dX", "dW"), actual, reference)
        },
    }
    g, r = actual[2], reference[2].float()
    du = (adam_direction(g) - adam_direction(r)).abs()
    index = int(du.flatten().argmax())
    row["Adam"] = {
        "max_direction_error": float(du.max()),
        "direction_rms": float(du.double().square().mean().sqrt()),
        "coordinates_over_0.01": int((du > 0.01).sum()),
        "worst_flat_index": index,
        "candidate_grad": float(g.flatten()[index]),
        "reference_grad": float(r.flatten()[index]),
        "max_gate_pass": float(du.max()) <= 0.01,
        "rms_gate_pass": float(du.double().square().mean().sqrt()) <= 0.001,
    }
    contract = radius_optimizer_contract()
    updated = []
    for grad in (g, r, r):
        p = torch.nn.Parameter(w.cuda().clone())
        opt = torch.optim.AdamW(
            [p],
            lr=contract["lr"],
            betas=contract["betas"],
            eps=contract["eps"],
            weight_decay=contract["dense_weight_decay"],
            fused=True,
        )
        p.grad = grad.cuda()
        opt.step()
        updated.append(
            (p.detach().cpu(), {k: v.detach().cpu() for k, v in opt.state[p].items()})
        )
    a, b, control = [v[0] for v in updated]
    delta = (a.double() - b.double()).abs()
    ulp = (torch.nextafter(b, torch.full_like(b, float("inf"))) - b).abs().double()
    row["Adam"]["post_w_max_abs"] = float(delta.max())
    row["Adam"]["post_w_bad_coordinates"] = int((delta > 1e-6 + 2 * ulp).sum())
    row["Adam"]["same_gradient_control_bytes"] = bool(
        torch.equal(
            b.contiguous().view(torch.uint8), control.contiguous().view(torch.uint8)
        )
    ) and all(
        torch.equal(
            updated[1][1][k].reshape(-1).view(torch.uint8),
            updated[2][1][k].reshape(-1).view(torch.uint8),
        )
        for k in updated[1][1]
    )
    return row


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    init = torch.load(
        args.inputs / "initialization.pt", map_location="cpu", weights_only=True
    )
    fixture = torch.load(
        args.inputs / "random-vjp.pt", map_location="cpu", weights_only=True
    )
    w, x, dy = init["weight"], fixture["x"], fixture["dy"]
    cases = [
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
    ]
    # Deliberately ill-conditioned sums, separately labelled from ordinary inputs.
    # Try all distinct K4 orders, without selecting a result after measurement.
    for i, values in enumerate(
        sorted(set(itertools.permutations((1.0, 1e-8, -1.0, 0.0))))
    ):
        dd = torch.zeros_like(dy)
        dd[:, 0] = torch.tensor(values)
        cases.append((f"constructed_cancellation_order{i:02}", torch.ones_like(x), dd))
    rows = []
    for name, xx, dd in cases:
        print(name, flush=True)
        rows.append(case(name, w, xx, dd))
        (args.output / "dense-control.json").write_text(
            json.dumps(
                {
                    "cases": rows,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(),
                    "TF32": False,
                    "scope": "FP32 GEMM versus independent CPU FP64; no timing or quality comparison",
                },
                indent=2,
            )
            + "\n"
        )
    assert all(r["Adam"]["same_gradient_control_bytes"] for r in rows)
    print(
        json.dumps(
            {
                "cases": len(rows),
                "ordinary_max_du": max(
                    r["Adam"]["max_direction_error"] for r in rows[:6]
                ),
                "constructed_max_du": max(
                    r["Adam"]["max_direction_error"] for r in rows[6:]
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
