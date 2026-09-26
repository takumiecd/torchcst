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
FUSED_CULL = FusedConfig(
    batch_rows=128, late_reduce=True, fp_fusion=False, cull_empty=True
)


def log(**data):
    print(json.dumps(data), flush=True)


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
            "fused_cull": asdict(FUSED_CULL),
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
    save(result)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    result["canonical"] = canonical
    save(result)
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
        "fused_cull_full": partial(
            layer, x, backend="triton_fused", fused_config=FUSED_CULL
        ),
        "fused_cull_prepared": partial(
            layer,
            x,
            backend="triton_fused",
            fused_config=FUSED_CULL,
            prepared=prepared,
        ),
    }
    result["checks"] = {}
    for name, fn in functions.items():
        validation = check(fn(), expected)
        result["checks"][name] = validation
        save(result)
        log(stage="validated", atoms=atoms, path=name, **validation)
    valid = {
        name: fn for name, fn in functions.items() if result["checks"][name]["passed"]
    }
    if valid:
        result.update(timing(valid))
    result["completed"] = True
    save(result)
    log(stage="timed", atoms=atoms, medians=result.get("median_ms", {}))


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

        run(args.size, args.batch, atoms, save)
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
