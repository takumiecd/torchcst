"""Compare internal execution schedules without changing chart tiles or precision.

python -m prototypes.benchmark_triton_schedule --output schedules.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from prototypes.benchmark_triton_linear import model, prepare


def probe(batch, atoms, split=False, rows=64, columns=128):
    from triton.testing import do_bench_cudagraph

    from torchcst.nn._backends._triton_kernels import (
        backward_atoms,
        backward_inputs,
        fused_forward,
        reduce_partials,
    )

    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    stations = (rows + 15) // 16
    with torch.no_grad():
        packed, circle, section, offsets = prepare(layer)
    x = torch.randn(batch, columns, device="cuda")
    dy = torch.randn(batch, rows, device="cuda")
    y, dx, dp = torch.empty_like(dy), torch.empty_like(x), torch.empty_like(packed)
    common = {
        "M": batch,
        "N": rows,
        "K": columns,
        "D": 4,
        "G": stations,
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
    if split:
        schedules = [(16, 16, 16, 8, 4)] * 5
    reference = None
    for index, (bm, bn, bk, ba, warps) in enumerate(schedules):
        automatic = split and index == 4
        parts_k = (1, 2, 4, 8, 1)[index] if split else 1
        parts_n = min(parts_k, stations)
        if automatic:
            from torchcst.nn._backends._schedule import split_count

            sm = torch.cuda.get_device_properties(x.device).multi_processor_count
            common_schedule = {
                "multiprocessors": sm,
                "atoms": atoms,
                "stations": stations,
            }
            parts_k = split_count(
                reduction_tiles=(columns + bk - 1) // bk,
                elements=y.numel(),
                programs=((batch + bm - 1) // bm) * stations,
                **common_schedule,
            )
            parts_n = split_count(
                reduction_tiles=stations,
                elements=dx.numel(),
                programs=((batch + bm - 1) // bm) * ((columns + bk - 1) // bk),
                **common_schedule,
            )
        partial_y = torch.empty((parts_k, *y.shape), device=x.device)
        partial_dx = torch.empty((parts_n, *dx.shape), device=x.device)
        options = dict(
            common, BM=bm, BN=bn, BK=bk, BA=ba, num_warps=warps, enable_fp_fusion=False
        )

        def forward(bm=bm, options=options, parts=parts_k, partial=partial_y):
            fused_forward[((batch + bm - 1) // bm, stations, parts)](
                x,
                packed,
                circle,
                section,
                offsets,
                y if parts == 1 else partial,
                **options,
                SPLIT_K=parts,
            )

            if parts > 1:
                reduce_partials[((y.numel() + 255) // 256,)](
                    partial, y, y.numel(), parts, 256
                )

        def inputs(bm=bm, bk=bk, options=options, parts=parts_n, partial=partial_dx):
            backward_inputs[((batch + bm - 1) // bm, (columns + bk - 1) // bk, parts)](
                dy,
                packed,
                circle,
                section,
                offsets,
                dx if parts == 1 else partial,
                **options,
                SPLIT_N=parts,
            )

            if parts > 1:
                reduce_partials[((dx.numel() + 255) // 256,)](
                    partial, dx, dx.numel(), parts, 256
                )

        def parameters(bk=bk, options=options):
            dp.zero_()
            backward_atoms[(stations, (columns + bk - 1) // bk)](
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
            "shape": [rows, columns],
            "bm": bm,
            "bn": bn,
            "bk": bk,
            "ba": ba,
            "warps": warps,
            "automatic": automatic,
            "split_k": parts_k,
            "split_n": parts_n,
            "max_errors": errors,
        }
        for name, fn in (("forward", forward), ("dx", inputs), ("dp", parameters)):
            row[name + "_gpu_ms"] = do_bench_cudagraph(fn, rep=20, return_mode="median")
        yield row


def main():
    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = Path("manifest.json")
    result = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else None,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cases": [],
    }
    cases = [
        (16, 64, 64, 128),
        (128, 64, 64, 128),
        (16, 256, 64, 128),
        (128, 256, 64, 128),
    ]
    if args.extended:
        cases += [(32, 512, 256, 512), (128, 512, 256, 512)]
    for batch, atoms, rows, columns in cases:
        for row in probe(batch, atoms, args.split, rows, columns):
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
