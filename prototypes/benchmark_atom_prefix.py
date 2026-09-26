"""A100 comparison of atom-prefix contraction, fused CST, and stored dense."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph

from prototypes.atom_prefix_linear import (
    atom_prefix_linear,
    contract_prefix,
    prepare_prefix,
)
from prototypes.benchmark_triton_linear import model


def error(actual, expected):
    return {
        "max_abs": (actual - expected).abs().max().item(),
        "relative_l2": (
            (actual - expected).norm() / expected.norm().clamp_min(1e-20)
        ).item(),
    }


def probe(batch, atoms, rows, columns, chunk):
    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    x = torch.randn(batch, columns, device="cuda", requires_grad=True)
    p = layer.atoms.p
    expected = F.linear(x, layer.dense_weight())
    actual = atom_prefix_linear(layer, x, p, atom_chunk=chunk)
    grad = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, p), grad)
    got = torch.autograd.grad(actual, (x, p), grad)
    errors = {
        "output": error(actual, expected),
        "dx": error(got[0], wanted[0]),
        "dp": error(got[1], wanted[1]),
    }
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(got[0], wanted[0], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(got[1], wanted[1], atol=1e-4, rtol=1e-4)
    del expected, actual, wanted, got
    with torch.no_grad():
        weight = layer.dense_weight().detach()
        prepared = prepare_prefix(layer, p)
        functions = {
            "stored_dense": lambda: F.linear(x, weight),
            "fused_cst": lambda: layer(x),
            "prefix_prepare": lambda: prepare_prefix(layer, p),
            "prefix_contract": lambda: contract_prefix(x, prepared, atom_chunk=chunk),
            "prefix_full": lambda: atom_prefix_linear(layer, x, p, atom_chunk=chunk),
        }
        # Interleave paths across rounds to expose timing variation.
        samples = {key: [] for key in functions}
        for round_index in range(3):
            order = (
                list(functions) if round_index % 2 == 0 else list(reversed(functions))
            )
            for key in order:
                samples[key].append(do_bench_cudagraph(functions[key], rep=30))
        peak = {}
        for key in ("stored_dense", "fused_cst", "prefix_full"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            result = functions[key]()
            torch.cuda.synchronize()
            peak[key] = torch.cuda.max_memory_allocated() - base
            del result
    return {
        "shape": [rows, columns],
        "batch": batch,
        "atoms": atoms,
        "atom_chunk": chunk,
        "errors": errors,
        "graph_ms_samples": samples,
        "graph_ms_median": {key: sorted(values)[1] for key, values in samples.items()},
        "forward_peak_extra_bytes": peak,
        "support_pairs": int(prepared.counts.sum().item()),
        "all_atom_site_pairs": atoms * rows * columns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--atom-chunk", type=int, default=64)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_commit": args.source_commit,
        "precision": "FP32; TF32 off",
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
        measured = probe(*case, args.atom_chunk)
        result["cases"].append(measured)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
