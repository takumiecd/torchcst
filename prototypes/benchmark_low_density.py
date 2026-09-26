"""Compare atom-dot and explicit fused routes below 5% atom density.

Dense W is the shared correctness oracle. Each atom count reuses one layer,
input, mapped weight, and prepared routing across full and prepared calls.
"""

import argparse
import gc
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_fused_config import FusedConfig
from prototypes.block_strip_linear import BlockStripLinear
from torchcst.nn._backends._preparation import prepare

FUSED = FusedConfig(batch_rows=128, late_reduce=True, fp_fusion=False)
BA1 = FusedConfig(128, 32, 16, 1, 4, True, False)


def log(**data):
    print(json.dumps(data), flush=True)


def empty_station_fraction(counts):
    """Fraction of stations whose support-layout buckets contain no atoms."""
    stations = (counts.numel() - 1) // 2
    if stations == 1:
        return float(counts[0] == 0)
    interior = counts[:-1:2]
    boundary = counts[1:-1:2]
    previous = torch.roll(boundary, 1)
    empty = (interior == 0) & (boundary == 0) & (previous == 0)
    return float(empty.sum().item()) / stations


@torch.no_grad()
def run(size, batch, atoms, save):
    torch.manual_seed(21)
    fraction = atoms / (size * size)
    result = {
        "size": size,
        "batch": batch,
        "atoms": atoms,
        "tile": [64, 64],
        "atom_fraction": fraction,
        "routes": {
            "atom_dot": {"backend": "triton_atom_dot", "BM": 64, "BN": 16},
            "fused": asdict(FUSED),
            "ba1": asdict(BA1),
        },
    }
    save(result)
    log(stage="initialize", size=size, batch=batch, atoms=atoms, atom_fraction=fraction)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    torch.cuda.synchronize()
    counts = torch.diff(prepared[3])
    assert int(counts.sum()) == atoms and bool((counts >= 0).all())
    result["bucket_summary"] = {
        "interior": int(counts[:-1:2].sum()),
        "boundary": int(counts[1:-1:2].sum()),
        "idle": int(counts[-1]),
        "max": int(counts.max()),
    }
    result["empty_station_fraction"] = empty_station_fraction(counts)
    save(result)
    try:
        weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    except AssertionError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        result["canonical"] = detail if isinstance(detail, dict) else {"passed": False}
        result["failure"] = "canonical dense W validation failed"
        result["completed"] = False
        save(result)
        log(
            stage="failed",
            atoms=atoms,
            failure=result["failure"],
            canonical=result["canonical"],
        )
        return False
    rows, cols = weight.shape
    if rows % 64 or cols % 64:
        raise RuntimeError("logical W must be divisible by 64")
    panels = weight.reshape(rows // 16, 16, cols // 16, 16)
    active_blocks = int((panels != 0).any(dim=(1, 3)).sum())
    total_blocks = (rows // 16) * (cols // 16)
    panels64 = weight.reshape(rows // 64, 64, cols // 64, 64)
    active_blocks64 = int((panels64 != 0).any(dim=(1, 3)).sum())
    total_blocks64 = (rows // 64) * (cols // 64)
    # Occupancy is measured only. Ultra-low density may leave panels empty; there is no tile culling.
    result["nonzero_weight_fraction"] = float((weight != 0).sum()) / weight.numel()
    result["active_16x16_blocks"] = active_blocks
    result["total_16x16_blocks"] = total_blocks
    result["active_64x64_blocks"] = active_blocks64
    result["total_64x64_blocks"] = total_blocks64
    result["canonical"] = canonical
    save(result)
    log(
        stage="weight_occupancy",
        atoms=atoms,
        nonzero_weight_fraction=result["nonzero_weight_fraction"],
        active_16x16_blocks=active_blocks,
        total_16x16_blocks=total_blocks,
        active_64x64_blocks=active_blocks64,
        total_64x64_blocks=total_blocks64,
        empty_station_fraction=result["empty_station_fraction"],
    )
    assert canonical["passed"], canonical
    x = torch.randn(batch, size, device="cuda")
    expected = F.linear(x, weight)
    functions = {
        "dense": partial(F.linear, x, weight),
        "atom_dot_full": partial(layer, x, backend="triton_atom_dot"),
        "atom_dot_prepared": partial(
            layer, x, backend="triton_atom_dot", prepared=prepared
        ),
        "fused_full": partial(layer, x, backend="triton_fused", fused_config=FUSED),
        "fused_prepared": partial(
            layer, x, backend="triton_fused", fused_config=FUSED, prepared=prepared
        ),
        "ba1_full": partial(layer, x, backend="triton_fused", fused_config=BA1),
        "ba1_prepared": partial(
            layer, x, backend="triton_fused", fused_config=BA1, prepared=prepared
        ),
    }
    result["checks"] = {}
    for name, fn in functions.items():
        validation = check(fn(), expected)
        result["checks"][name] = validation
        save(result)
        log(stage="validated", atoms=atoms, path=name, **validation)
        if not validation["passed"]:
            result["failure"] = f"{name} validation failed"
            result["completed"] = False
            save(result)
            log(stage="failed", atoms=atoms, failure=result["failure"], path=name)
            return False
    valid = {
        name: fn for name, fn in functions.items() if result["checks"][name]["passed"]
    }
    if valid:
        result.update(timing(valid))
    result["completed"] = True
    save(result)
    log(stage="timed", atoms=atoms, medians=result.get("median_ms", {}))
    return True


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument(
        "--atoms", nargs="+", type=int, default=[8192, 32768, 65536, 131072]
    )
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.source_commit
    ):
        parser.error("source commit must be full 40-character hex")
    if args.size < 64 or args.size % 64 or args.batch < 1:
        parser.error("positive batch and size divisible by 64 required")
    dense = args.size * args.size
    if any(atoms < 1 or atoms / dense >= 0.05 for atoms in args.atoms):
        parser.error("each atom count must be positive and below 5% of N*K")
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
        "seed": 21,
        "timing": "CUDA Graph median of 3 rounds rep=20ms; dense W precomputed",
        "cases": [],
    }
    for atoms in args.atoms:
        result["cases"].append({})

        def save(case):
            result["cases"][-1] = case
            (args.output_dir / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

        if not run(args.size, args.batch, atoms, save):
            result["completed"] = False
            result["failure"] = (
                "route validation failed; sweep stopped before lower atom counts"
            )
            (args.output_dir / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            raise SystemExit(result["failure"])
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
