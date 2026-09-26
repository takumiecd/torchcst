"""Compare direct input reuse at identical geometry, atoms, and dense weights."""

import argparse
import gc
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check, peak_extra
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_kernels import block_direct_reuse
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture, resource_record
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def run(size, batch, atoms, output_dir, profile):
    torch.manual_seed(21)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared)
    x = torch.randn(batch, size, device="cuda")
    variants = {
        "old_bm16": ("triton_direct", 16),
        "old_bm64": ("triton_direct", 64),
        "reuse_bm16": ("triton_reuse", 16),
        "reuse_bm64": ("triton_reuse", 64),
        "reuse_bm128": ("triton_reuse", 128),
        "fused_bm64": ("triton_fused", 64),
    }
    functions = {"dense": partial(F.linear, x, weight)}
    functions.update(
        {
            name: partial(layer, x, backend=backend, batch_tile=bm)
            for name, (backend, bm) in variants.items()
        }
    )
    expected = functions["dense"]()
    checks = {name: check(fn(), expected) for name, fn in functions.items()}
    assert canonical["passed"] and all(v["passed"] for v in checks.values()), checks
    peaks = {name: peak_extra(fn) for name, fn in functions.items()}
    result = {
        "size": size,
        "batch": batch,
        "atoms": atoms,
        "canonical": canonical,
        "checks": checks,
        "extra_peak_bytes": peaks,
        "bucket_counts": torch.diff(prepared[3]).cpu().tolist(),
        **timing(functions),
    }
    packed, circle, section, offsets = prepared
    bounds = torch.stack((section.amin(0), section.amax(0)))
    y = torch.empty_like(expected)
    result["resources"] = {}
    for bm in (16, 64, 128):
        grid = (triton.cdiv(batch, bm), size)
        compiled = block_direct_reuse[grid](
            x,
            packed,
            circle,
            section,
            offsets,
            bounds,
            y,
            M=batch,
            N=size,
            K=size,
            S=64,
            T=64,
            CG=size // 64,
            G=(size // 64) ** 2,
            D=4,
            PROFILE=1,
            BM=bm,
            BK=64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        result["resources"][str(bm)] = resource_record(compiled, grid)
    if profile:
        view = SimpleNamespace(out_features=size, atoms=layer.strip.atoms)
        result["profiles"] = [
            capture(view, x, name, functions[name], False, output_dir)
            for name in ("old_bm16", "reuse_bm64", "reuse_bm128")
        ]
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["4096:128:8192", "4096:512:8192", "8192:128:16384", "4096:128:131072"],
    )
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.source_commit
    ):
        parser.error("source commit must be 40 hex characters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "precision": "FP32 TF32 off",
        "tile": [64, 64],
        "timing": "CUDA Graph, preparation included, median 3 rounds rep=20ms",
        "cases": [],
    }
    for index, spec in enumerate(args.cases):
        size, batch, atoms = map(int, spec.split(":"))
        print(json.dumps({"stage": "start", "case": spec}), flush=True)
        result["cases"].append(run(size, batch, atoms, args.output_dir, index == 0))
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "stage": "case_done",
                    "case": spec,
                    "medians": result["cases"][-1]["median_ms"],
                }
            ),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
