"""Compare internal execution schedules without changing chart tiles or precision.

python -m prototypes.benchmark_triton_schedule --output schedules.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prototypes.benchmark_triton_linear import model, prepare


def probe(batch, atoms):
    from triton.testing import do_bench_cudagraph

    from torchcst.nn._backends._triton_kernels import (
        backward_atoms,
        backward_inputs,
        fused_forward,
    )

    torch.manual_seed(21)
    layer = model(atoms)
    with torch.no_grad():
        packed, circle, section, offsets = prepare(layer)
    x = torch.randn(batch, 128, device="cuda")
    dy = torch.randn(batch, 64, device="cuda")
    y, dx, dp = torch.empty_like(dy), torch.empty_like(x), torch.empty_like(packed)
    common = {
        "M": batch,
        "N": 64,
        "K": 128,
        "D": 4,
        "G": 4,
        "STATION_ROWS": 16,
        "PROFILE": 1,
    }
    schedules = [(16, 16, 16, 8, 4)] + [
        (bm, 16, bk, ba, 4)
        for bm, bk, ba in (
            (16, 16, 32),
            (16, 16, 64),
            (16, 32, 32),
            (32, 16, 32),
            (32, 32, 32),
            (16, 32, 64),
        )
    ]
    reference = None
    results = []
    for bm, bn, bk, ba, warps in schedules:
        options = dict(
            common, BM=bm, BN=bn, BK=bk, BA=ba, num_warps=warps, enable_fp_fusion=False
        )

        def forward(bm=bm, options=options):
            fused_forward[((batch + bm - 1) // bm, 4)](
                x, packed, circle, section, offsets, y, **options
            )

        def inputs(bm=bm, bk=bk, options=options):
            backward_inputs[((batch + bm - 1) // bm, (128 + bk - 1) // bk)](
                dy, packed, circle, section, offsets, dx, **options
            )

        def parameters(bk=bk, options=options):
            dp.zero_()
            backward_atoms[(4, (128 + bk - 1) // bk)](
                x, dy, packed, circle, section, offsets, dp, **options
            )

        forward()
        inputs()
        parameters()
        if reference is None:
            reference = [t.clone() for t in (y, dx, dp)]
        errors = []
        for actual, expected in zip((y, dx, dp), reference):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            errors.append((actual - expected).abs().max().item())
        row = {
            "batch": batch,
            "atoms": atoms,
            "bm": bm,
            "bn": bn,
            "bk": bk,
            "ba": ba,
            "warps": warps,
            "max_errors": errors,
        }
        for name, fn in (("forward", forward), ("dx", inputs), ("dp", parameters)):
            row[name + "_gpu_ms"] = do_bench_cudagraph(fn, rep=20, return_mode="median")
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def main():
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cases": [],
    }
    for batch, atoms in ((16, 64), (128, 64), (16, 256), (128, 256)):
        result["cases"].extend(probe(batch, atoms))
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
