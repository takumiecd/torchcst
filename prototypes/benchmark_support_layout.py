"""Measure dependency packing and reuse across input rows on A100."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from prototypes.benchmark_triton_linear import model
from torchcst.nn._backends._preparation import prepare
from torchcst.nn._backends._triton import _FusedLinear, forward
from torchcst.nn._support_layout import station_buckets


def probe(batch, atoms, rows, columns):
    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    x = torch.randn(batch, columns, device="cuda", requires_grad=True)
    p = layer.atoms.p
    expected = F.linear(x, layer.dense_weight())
    grad = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, p), grad)
    errors = {}
    for bm in (16, 32, 64):
        result = forward(layer, x, p, batch_tile=bm)
        got = torch.autograd.grad(result, (x, p), grad)
        torch.testing.assert_close(result, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got[0], wanted[0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got[1], wanted[1], atol=1e-4, rtol=1e-4)
        errors[bm] = [
            float((a - b).abs().max())
            for a, b in zip((result, *got), (expected, *wanted), strict=True)
        ]
    del result, got, expected, wanted
    with torch.no_grad():
        old = prepare(layer, p)
        new = prepare(layer, p, support_layout=True)
        stations = layer.chart.tile_count
        counts = torch.diff(new[-1]).cpu().tolist()
        old_counts = torch.diff(old[-1]).cpu().tolist()
        candidates = [
            sum(counts[b] for b in station_buckets(g, stations))
            for g in range(stations)
        ]
        old_candidates = [
            sum(old_counts[j] for j in {(g - 1) % stations, g, (g + 1) % stations})
            for g in range(stations)
        ]
        weight = layer.dense_weight().detach()
        functions = {
            "dense": lambda: F.linear(x, weight),
            "old_full": lambda: forward(layer, x, p, support_layout=False),
            "old_prepared": lambda: _FusedLinear.apply(x, *old, 16, 1),
            "new_prepare": lambda: prepare(layer, p, support_layout=True),
        }
        for bm in (16, 32, 64):
            functions[f"new_full_bm{bm}"] = lambda bm=bm: forward(
                layer, x, p, batch_tile=bm
            )
            functions[f"new_prepared_bm{bm}"] = lambda bm=bm: _FusedLinear.apply(
                x, *new, 16, 1, False, True, bm
            )
        samples = {name: [] for name in functions}
        for round_index in range(3):
            names = (
                list(functions) if round_index % 2 == 0 else list(reversed(functions))
            )
            for name in names:
                samples[name].append(do_bench_cudagraph(functions[name], rep=30))
        peaks = {}
        for name in ("old_full", "new_full_bm16", "new_full_bm32", "new_full_bm64"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            output = functions[name]()
            torch.cuda.synchronize()
            peaks[name] = torch.cuda.max_memory_allocated() - base
            del output
    return {
        "shape": [rows, columns],
        "batch": batch,
        "atoms": atoms,
        "errors": errors,
        "bucket_counts": counts,
        "old_candidates": old_candidates,
        "new_candidates": candidates,
        "graph_ms": {name: median(values) for name, values in samples.items()},
        "samples": samples,
        "peak_extra_bytes": peaks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_commit": args.source_commit,
        "cases": [],
    }
    for case in (
        (16, 64, 64, 128),
        (128, 64, 64, 128),
        (16, 256, 64, 128),
        (128, 256, 64, 128),
        (32, 512, 256, 512),
        (128, 512, 256, 512),
    ):
        result["cases"].append(probe(*case))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
