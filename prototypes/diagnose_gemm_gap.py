"""Isolate preparation, generated weights, and GEMM implementation overhead."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl
from triton.testing import do_bench_cudagraph

from prototypes.benchmark_triton_linear import model
from torchcst.nn._backends._preparation import prepare
from torchcst.nn._backends._triton_kernels import fused_forward, materialize_weights


@tr.jit
def cached_prototype(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """Same 16x16 IEEE dot loop as production, but load a cached dense weight."""
    m = tl.program_id(0) * 16 + tl.arange(0, 16)
    n = tl.program_id(1) * 16 + tl.arange(0, 16)
    acc = tl.full((16, 16), 0.0, tl.float32)
    for start in range(0, K, 16):
        k = start + tl.arange(0, 16)
        x = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.0
        )
        w = tl.load(
            W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0.0
        )
        acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def resource_info(compiled):
    return {
        "registers_per_thread": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_bytes": compiled.metadata.shared,
    }


@torch.no_grad()
def probe(batch, atoms, rows, columns):
    torch.manual_seed(21)
    layer = model(atoms, rows=rows, columns=columns)
    packed, circle, section, offsets = prepare(layer, layer.atoms.p)
    x = torch.randn(batch, columns, device="cuda")
    y = torch.empty(batch, rows, device="cuda")
    cached = torch.empty_like(y)
    weight = torch.empty(rows, columns, device="cuda")
    stations = rows // 16
    options = {
        "N": rows,
        "K": columns,
        "D": packed.shape[1] - 2,
        "G": stations,
        "STATION_ROWS": 16,
        "PROFILE": 1,
        "BN": 16,
        "BK": 16,
        "BA": 8,
        "num_warps": 4,
        "enable_fp_fusion": False,
    }

    def generate():
        return materialize_weights[(stations, tr.cdiv(columns, 16))](
            packed, circle, section, offsets, weight, **options
        )

    def fused():
        return fused_forward[(tr.cdiv(batch, 16), stations)](
            x, packed, circle, section, offsets, y, M=batch, BM=16, **options
        )

    def prototype():
        return cached_prototype[(tr.cdiv(batch, 16), stations)](
            x,
            weight,
            cached,
            batch,
            rows,
            columns,
            num_warps=4,
            enable_fp_fusion=False,
        )

    def generated_then_vendor():
        generate()
        return F.linear(x, weight)

    generator_info = resource_info(generate())
    fused_info = resource_info(fused())
    prototype_info = resource_info(prototype())
    reference = F.linear(x, weight)
    torch.testing.assert_close(y, reference, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cached, reference, atol=3e-5, rtol=3e-5)
    functions = {
        "prepare": lambda: prepare(layer, layer.atoms.p),
        "generate_weight_once": generate,
        "cached_vendor_gemm": lambda: F.linear(x, weight),
        "cached_prototype_gemm": prototype,
        "generate_then_vendor": generated_then_vendor,
        "prepared_fused": fused,
        "full_cst": lambda: layer(x),
    }
    timings = {
        name: do_bench_cudagraph(fn, rep=30, return_mode="median")
        for name, fn in functions.items()
    }
    owners = torch.diff(offsets).cpu().tolist()
    candidate_counts = [
        sum(owners[j] for j in {(g - 1) % stations, g, (g + 1) % stations})
        for g in range(stations)
    ]
    return {
        "batch": batch,
        "atoms": atoms,
        "shape": [rows, columns],
        "parameter_count": {"dense": rows * columns, "cst": layer.atoms.p.numel()},
        "owner_counts": owners,
        "candidate_atoms_per_station": candidate_counts,
        "weight_generation_repetitions": tr.cdiv(batch, 16),
        "graph_ms": timings,
        "compiled_resources": {
            "generate": generator_info,
            "fused": fused_info,
            "cached_prototype": prototype_info,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": tr.__version__,
        "source_commit": args.source_commit,
        "precision": "FP32 IEEE; TF32 off",
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
        measured = probe(*case)
        result["cases"].append(measured)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(measured), flush=True)


if __name__ == "__main__":
    main()
