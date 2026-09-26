"""Prepared-only timing of the fused stage kernels and the existing fused kernel."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_fused_config import FusedConfig, launch_fused
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.fused_stage_kernels import fused_stage
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare

BM, BN, BK, BA = 128, 16, 16, 8


def log(**value):
    print(json.dumps(value), flush=True)


def launch(grid, tensors, consts, stage, y, sink, w):
    x, p, circle, section, offsets = tensors
    return fused_stage[grid](
        x,
        p,
        circle,
        section,
        offsets,
        y,
        sink,
        w,
        **consts,
        STAGE=stage,
        num_warps=4,
        enable_fp_fusion=False,
    )


def compiled_record(compiled, output_dir, stem):
    ptx = compiled.asm["ptx"]
    path = output_dir / f"{stem}.ptx"
    path.write_text(ptx)
    return {
        "n_regs": compiled.n_regs,
        "n_spills": compiled.n_spills,
        "shared": compiled.metadata.shared,
        "ptx_path": str(path),
        "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--atoms", type=int, default=3355443)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    assert args.size % 64 == 0 and min(args.size, args.batch, args.atoms) > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(21)
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "size": args.size,
        "batch": args.batch,
        "atoms": args.atoms,
        "config": {"BM": BM, "BN": BN, "BK": BK, "BA": BA, "num_warps": 4},
        "timing_note": (
            "stage1 writes checksum Sink and stage2 uses one synthetic "
            "runtime 16x16 W fragment reused across tile iterations; their "
            "times are counterfactual and non-additive."
        ),
    }
    path = args.output_dir / "results.json"

    def save():
        path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    layer = BlockStripLinear(
        (args.size, args.size), (64, 64), args.atoms, device="cuda"
    )
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    result["canonical"] = canonical
    assert canonical["passed"]
    x = torch.randn(args.batch, args.size, device="cuda")
    expected = F.linear(x, weight)
    cfg = FusedConfig(128, 16, 16, 8, 4, True)
    grid = (
        math.ceil(args.batch / BM),
        layer.row_groups * math.ceil(64 / BN),
    )
    p, circle, section, offsets = prepared
    consts = {
        "M": args.batch,
        "N": args.size,
        "K": args.size,
        "S": 64,
        "T": 64,
        "CG": layer.column_groups,
        "G": layer.strip.chart.tile_count,
        "D": p.shape[1] - 2,
        "PROFILE": PROFILE_KINDS[type(layer.strip.kernel.profile)],
        "BM": BM,
        "BN": BN,
        "BK": BK,
        "BA": BA,
    }
    tensors = (x, p, circle, section, offsets)
    y_fused = torch.empty(args.batch, args.size, device="cuda")
    compiled_existing = launch_fused(layer, x, prepared, y_fused, cfg)
    y0 = torch.empty_like(y_fused)
    sink = torch.empty(grid[0] * grid[1], device="cuda")
    w = torch.randn(16, 16, device="cuda")
    compiled = launch(grid, tensors, consts, 0, y0, sink, w)
    result["stages"] = {
        "0": compiled_record(compiled, args.output_dir, "stage0"),
        "existing_fused": compiled_record(
            compiled_existing, args.output_dir, "existing_fused"
        ),
    }
    result["checks"] = {
        "stage0_vs_fused": check(y0, y_fused),
        "stage0_vs_dense": check(y0, expected),
    }
    save()
    log(stage="checks", **result["checks"])
    assert result["checks"]["stage0_vs_fused"]["passed"]
    assert result["checks"]["stage0_vs_dense"]["passed"]
    y1 = torch.empty_like(y0)
    for stage, output in ((1, y1), (2, y1)):
        compiled = launch(grid, tensors, consts, stage, output, sink, w)
        result["stages"][str(stage)] = compiled_record(
            compiled, args.output_dir, f"stage{stage}"
        )
        save()
    result.update(
        timing(
            {
                "stage0": lambda: launch(grid, tensors, consts, 0, y0, sink, w),
                "stage1": lambda: launch(grid, tensors, consts, 1, y1, sink, w),
                "stage2": lambda: launch(grid, tensors, consts, 2, y1, sink, w),
                "existing_fused": lambda: launch_fused(
                    layer, x, prepared, y_fused, cfg
                ),
            }
        )
    )
    result["completed"] = True
    save()
    log(stage="completed", median_ms=result["median_ms"])


if __name__ == "__main__":
    main()
