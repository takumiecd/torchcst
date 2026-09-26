"""Measure support sparsity and the cost of skipping column fragments."""

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
from prototypes.block_shared_kernel import block_direct_shared
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_support import section_bounds
from prototypes.block_support_diagnostics import diagnose_support
from prototypes.profile_current_paths import capture, resource_record
from torchcst.nn._backends._preparation import prepare


@torch.no_grad()
def run(size, batch, atoms, output_dir, profile, configs):
    torch.manual_seed(21)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared)
    x = torch.randn(batch, size, device="cuda")
    functions = {
        "dense": partial(F.linear, x, weight),
        "fused64": partial(layer, x, backend="triton_fused", batch_tile=64),
    }
    for spec in configs:
        width, mode = spec.split(":")
        functions[spec] = partial(
            layer,
            x,
            backend="triton_atom_dot",
            column_tile=int(width),
            support_cull=mode,
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
        "support": diagnose_support(layer, prepared, batch),
        "checks": checks,
        "extra_peak_bytes": peaks,
        "bucket_counts": torch.diff(prepared[3]).cpu().tolist(),
        **timing(functions),
    }
    packed, circle, section, offsets = prepared
    bounds = torch.stack((section.amin(0), section.amax(0)))
    y = torch.empty_like(expected)
    result["resources"] = {}
    for spec in configs:
        width, mode = spec.split(":")
        width = int(width)
        bounds = (
            section_bounds(section, width)
            if mode != "none"
            else torch.stack((section.amin(0), section.amax(0)))
        )
        grid = (triton.cdiv(batch, 64), size // 16)
        compiled = block_direct_shared[grid](
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
            BM=64,
            BN=16,
            BK=width,
            USE_DOT=True,
            CULL_CHUNKS=mode != "none",
            CHECK_ZERO=mode == "exact",
            num_warps=4,
            enable_fp_fusion=False,
        )
        result["resources"][spec] = resource_record(compiled, grid)
    if profile:
        view = SimpleNamespace(out_features=size, atoms=layer.strip.atoms)
        fastest = min(configs, key=lambda key: result["median_ms"][key])
        result["profiles"] = [
            capture(view, x, name.replace(":", "_"), functions[name], False, output_dir)
            for name in dict.fromkeys(("64:none", fastest))
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
        default=["4096:128:8192"],
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "64:none",
            "16:none",
            "16:bounds",
            "16:exact",
            "32:none",
            "32:bounds",
            "32:exact",
            "64:exact",
        ],
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
        result["cases"].append(
            run(
                size,
                batch,
                atoms,
                args.output_dir,
                index == 0,
                args.configs,
            )
        )
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
