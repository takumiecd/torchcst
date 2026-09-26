"""Remeasure compact CST at atoms = round(dense elements * 0.05).

Dense W is a correctness/control artifact only. Both CST paths retain bounded
local intermediates. Prepared calls separate repeated routing from contraction.
"""

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
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import prepare


def log(**data):
    print(json.dumps(data), flush=True)


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


@torch.no_grad()
def run(size, batch, output_dir, save):
    atoms = round(size * size * 0.05)
    torch.manual_seed(21)
    result = {"size": size, "batch": batch, "atoms": atoms}
    save(result)
    log(stage="initialize", **result)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepare_fn = partial(prepare, layer.strip, layer.strip.atoms.p, support_layout=True)
    prepared = prepare_fn()
    torch.cuda.synchronize()
    counts = torch.diff(prepared[3])
    assert int(counts.sum()) == atoms and bool((counts >= 0).all())
    result.update(
        atoms_per_tile=atoms / layer.strip.chart.tile_count,
        tile_count=layer.strip.chart.tile_count,
        atom_parameter_shape=list(layer.strip.atoms.p.shape),
        parameter_bytes=tensor_bytes(layer.strip.atoms.p),
        prepared_tensor_bytes=[tensor_bytes(t) for t in prepared],
        prepared_bytes=sum(tensor_bytes(t) for t in prepared),
        dense_weight_bytes=size * size * 4,
        bucket_summary={
            "interior": int(counts[:-1:2].sum()),
            "boundary": int(counts[1:-1:2].sum()),
            "idle": int(counts[-1]),
            "max": int(counts.max()),
        },
    )
    save(result)
    log(stage="oracle", size=size, atoms=atoms)
    # All atoms participate at each of the same 1152 sites as previous studies.
    # Bound temporary geometry storage at the new, much larger atom count.
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    result["canonical"] = {**canonical, "sites": 1152, "chunk": 4}
    save(result)
    assert canonical["passed"], canonical
    x = torch.randn(batch, size, device="cuda")
    functions = {"dense": partial(F.linear, x, weight)}
    for name, backend in (("atom_dot", "triton_atom_dot"), ("fused", "triton_fused")):
        functions[name + "_full"] = partial(layer, x, backend=backend, batch_tile=64)
        functions[name + "_prepared"] = partial(
            layer, x, backend=backend, batch_tile=64, prepared=prepared
        )
    expected = functions["dense"]()
    result["checks"] = {}
    for name, fn in functions.items():
        validation = check(fn(), expected)
        result["checks"][name] = validation
        save(result)
        log(stage="validated", size=size, path=name, **validation)
    # Preserve failures in results; do not present a failing path as valid speed.
    valid = {
        name: fn for name, fn in functions.items() if result["checks"][name]["passed"]
    }
    valid["prepare_only"] = prepare_fn
    result["extra_peak_bytes"] = {}
    for name, fn in valid.items():
        result["extra_peak_bytes"][name] = peak_extra(fn)
        save(result)
    result.update(input_bytes=tensor_bytes(x), output_bytes=tensor_bytes(expected))
    result.update(timing(valid))
    save(result)
    log(stage="timed", size=size, medians=result["median_ms"])
    view = SimpleNamespace(out_features=size, atoms=layer.strip.atoms)
    result["profiles"] = []
    for name in ("atom_dot_full", "fused_full", "prepare_only"):
        if name in valid:
            result["profiles"].append(
                capture(view, x, name, valid[name], False, output_dir)
            )
            save(result)
    result["completed"] = True
    save(result)
    log(stage="completed", size=size)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096])
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.source_commit
    ):
        parser.error("source commit must be full 40-character hex")
    if args.batch < 1 or any(s < 64 or s % 64 for s in args.sizes):
        parser.error("positive batch and sizes divisible by 64 required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name, prop.name
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "precision": "FP32 TF32 off",
        "tile": [64, 64],
        "atom_policy": "round(N*K*0.05), atom count, not scalar parameter count",
        "timing": "CUDA Graph median of 3 rounds rep=20ms; dense W precomputed",
        "cases": [],
    }
    for size in args.sizes:
        case_dir = args.output_dir / str(size)
        case_dir.mkdir(exist_ok=True)
        result["cases"].append({})

        def save(case):
            result["cases"][-1] = case
            (args.output_dir / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

        run(size, args.batch, case_dir, save)
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
