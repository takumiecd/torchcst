"""Compare split-K mapped GEMM against the same 5% atoms and dense weight."""

import argparse
import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_fused_config import FusedConfig
from prototypes.block_strip_kernels import block_fused, block_split_reduce
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import PROFILE_KINDS, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--splits", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--stages", type=int, nargs="+", default=[])
    parser.add_argument("--column-factored", action="store_true")
    parser.add_argument("--cull-tile", action="store_true")
    parser.add_argument("--radial-factor", action="store_true")
    args = parser.parse_args()
    assert args.size % 64 == 0 and args.batch > 0
    assert all(split in (2, 4, 8, 16) for split in args.splits)
    assert all(stage in (1, 2, 3, 4, 5) for stage in args.stages)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    assert "A100" in props.name
    atoms = round(args.size * args.size * 0.05)
    layer = BlockStripLinear((args.size, args.size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"]
    x = torch.randn(args.batch, args.size, device="cuda")
    expected = F.linear(x, weight)
    cfg = FusedConfig(128, 16, 32, 1, 4, True, True)
    legacy_cfg = FusedConfig(128, 16, 16, 8, 4, True, True)

    def split_forward(
        split,
        hoist_section=False,
        num_stages=None,
        column_factored=False,
        cull_tile=False,
        radial_factor=False,
    ):
        p, circle, section, offsets = (
            prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
            if args.full
            else prepared
        )
        y = torch.empty_like(expected)
        partial = torch.empty((split, *expected.shape), device="cuda")
        s, t = layer.tile_shape
        launch_options = {"num_warps": cfg.warps, "enable_fp_fusion": cfg.fp_fusion}
        if num_stages is not None:
            launch_options["num_stages"] = num_stages
        block_fused[
            (
                math.ceil(args.batch / cfg.batch_rows),
                layer.row_groups * math.ceil(s / cfg.output_rows),
                split,
            )
        ](
            x,
            p,
            circle,
            section,
            offsets,
            partial,
            M=args.batch,
            N=args.size,
            K=args.size,
            S=s,
            T=t,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            D=p.shape[1] - 2,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            BM=cfg.batch_rows,
            BN=cfg.output_rows,
            BK=cfg.columns,
            BA=cfg.atoms,
            LATE_REDUCE=cfg.late_reduce,
            SPLIT_K=split,
            HOIST_SECTION=hoist_section,
            COLUMN_FACTORED=column_factored,
            CULL_TILE=cull_tile,
            RADIAL_FACTOR=radial_factor,
            **launch_options,
        )
        block_split_reduce[(triton.cdiv(y.numel(), 256),)](
            partial, y, args.batch, args.size, split
        )
        return y

    functions = {
        "dense": partial(F.linear, x, weight),
        "legacy": partial(
            layer,
            x,
            backend="triton_fused",
            fused_config=legacy_cfg,
            prepared=None if args.full else prepared,
        ),
        "baseline": partial(
            layer,
            x,
            backend="triton_fused",
            fused_config=cfg,
            prepared=None if args.full else prepared,
        ),
        "default": partial(
            layer,
            x,
            backend="triton_fused",
            prepared=None if args.full else prepared,
        ),
    }
    functions.update({f"split_{n}": partial(split_forward, n) for n in args.splits})
    if args.column_factored:
        functions.update(
            {
                f"column_factored_{n}": partial(split_forward, n, True, None, True)
                for n in args.splits
            }
        )
    if 8 in args.splits:
        functions["split_8_hoist"] = partial(split_forward, 8, True)
        functions.update(
            {
                f"stages_{stage}": partial(split_forward, 8, True, stage)
                for stage in args.stages
            }
        )
        if args.cull_tile:
            functions["cull_tile"] = partial(split_forward, 8, True, None, True, True)
        if args.radial_factor:
            functions["radial_factor"] = partial(
                split_forward, 8, True, None, True, False, True
            )
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert all(value["passed"] for value in checks.values()), checks
    results = {
        "source_commit": args.source_commit,
        "device": props.name,
        "multiprocessors": props.multi_processor_count,
        "shape": [args.batch, args.size, args.size],
        "atoms": atoms,
        "atom_fraction": atoms / (args.size * args.size),
        "full": args.full,
        "config": asdict(cfg),
        "canonical": canonical,
        "checks": checks,
        "timing": "CUDA Graph 3 rounds rep=20ms; dense W precomputed",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "median_ms": results["median_ms"]}))


if __name__ == "__main__":
    main()
