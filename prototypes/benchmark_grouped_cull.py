"""Prepared-only paired A100 comparison for exact group-8 culling at 5%."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_fused_config import FusedConfig
from prototypes.block_strip_kernels import block_split_reduce
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.grouped_cull_kernels import block_grouped_fused, build_grouped
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    if args.size != 4096:
        raise ValueError("this bounded pilot targets 4096x4096")
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name:
        raise RuntimeError("this pilot requires A100")
    size, batch = args.size, 128
    atoms = round(size * size * 0.05)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    grouped = build_grouped(prepared)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"], canonical
    x = torch.randn(batch, size, device="cuda")
    expected = F.linear(x, weight)
    cfg16 = FusedConfig(128, 16, 32, 1, 4, True, True, 8, True, True)
    cfg32 = FusedConfig(128, 32, 32, 1, 4, True, True, 8, True, True)
    resources = {}

    def grouped_forward():
        partial = x.new_empty((8, batch, size))
        result = x.new_empty((batch, size))
        compiled = block_grouped_fused[(1, (size // 64) * 4, 8)](
            x,
            grouped.packed,
            grouped.circle,
            grouped.section,
            grouped.group_offsets,
            grouped.group_starts,
            grouped.group_counts,
            grouped.group_bounds,
            grouped.site_bounds,
            partial,
            M=batch,
            N=size,
            K=size,
            S=64,
            T=64,
            CG=size // 64,
            G=(size // 64) ** 2,
            PROFILE=1,
            BM=128,
            BN=16,
            BK=32,
            SPLIT_K=8,
            num_warps=4,
            enable_fp_fusion=True,
        )
        if not resources:
            resources.update(
                registers_per_thread=compiled.n_regs,
                compiler_spills=compiled.n_spills,
                shared_bytes_per_block=compiled.metadata.shared,
            )
        block_split_reduce[(triton.cdiv(result.numel(), 256),)](
            partial, result, batch, size, 8
        )
        return result

    functions = {
        "dense_precomputed": partial(F.linear, x, weight),
        "baseline_16": partial(
            layer, x, backend="triton_fused", prepared=prepared, fused_config=cfg16
        ),
        "baseline_32": partial(
            layer, x, backend="triton_fused", prepared=prepared, fused_config=cfg32
        ),
        "grouped_16": grouped_forward,
    }
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    if not all(value["passed"] for value in checks.values()):
        raise AssertionError(checks)
    measured = timing(functions)
    memory = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            grouped.packed,
            grouped.group_offsets,
            grouped.group_starts,
            grouped.group_counts,
            grouped.group_bounds,
            grouped.site_bounds,
        )
    )
    result = {
        "source_commit": args.source_commit,
        "device": props.name,
        "multiprocessors": props.multi_processor_count,
        "shape": [batch, size, size],
        "atoms": atoms,
        "atom_fraction": atoms / (size * size),
        "mode": "prepared only; group builder outside timed functions",
        "groups": grouped.groups,
        "group_build_host_seconds": grouped.build_seconds,
        "grouped_tensor_payload_bytes": memory,
        "resources": resources,
        "canonical": canonical,
        "checks": checks,
        "timing": "CUDA Graph 3 rounds rep=20ms; dense W and grouping precomputed",
        **measured,
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "median_ms": measured["median_ms"]}))


if __name__ == "__main__":
    main()
