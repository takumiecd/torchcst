"""Paired exhaustive/local routing at the target five-percent atom density."""

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check, peak_extra
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import execution_plan, prepare


def log(**value):
    print(json.dumps(value), flush=True)


@torch.no_grad()
def run(size, out, save):
    torch.manual_seed(21)
    atoms, batch = round(size * size * 0.05), 128
    result = {"size": size, "atoms": atoms, "batch": batch}
    save(result)
    log(stage="start", **result)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    site = layer.strip
    local_plan = execution_plan(site)
    assert local_plan.routing.local_candidates
    old_plan = replace(
        local_plan, routing=replace(local_plan.routing, local_candidates=False)
    )

    def with_plan(plan, fn):
        # Benchmark-only selection: same fixed tensors and version key, with
        # only the local/exhaustive dispatch bit changed. No GPU tensor mutation.
        previous = site._strip_torus_plan
        site._strip_torus_plan = plan
        try:
            return fn()
        finally:
            site._strip_torus_plan = previous

    def prep(plan):
        return with_plan(plan, lambda: prepare(site, site.atoms.p, support_layout=True))

    prepared = prep(local_plan)
    old_prepared = prep(old_plan)
    for got, wanted in zip(prepared, old_prepared):
        torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    del old_prepared
    result["prepared_exact_parity"] = True
    counts = torch.diff(prepared[3])
    result["bucket_summary"] = {
        "interior": int(counts[:-1:2].sum()),
        "boundary": int(counts[1:-1:2].sum()),
        "idle": int(counts[-1]),
    }
    save(result)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    result["canonical"] = canonical
    x = torch.randn(batch, size, device="cuda")
    expected = F.linear(x, weight)
    functions = {
        "dense": lambda: F.linear(x, weight),
        "prepare_exhaustive": lambda: prep(old_plan),
        "prepare_local": lambda: prep(local_plan),
        "fused_exhaustive": lambda: with_plan(
            old_plan, lambda: layer(x, backend="triton_fused")
        ),
        "fused_local": lambda: with_plan(
            local_plan, lambda: layer(x, backend="triton_fused")
        ),
        "fused_prepared": lambda: layer(x, backend="triton_fused", prepared=prepared),
    }
    result["checks"] = {}
    for name, fn in functions.items():
        if not name.startswith("prepare_"):
            result["checks"][name] = check(fn(), expected)
            assert result["checks"][name]["passed"], result["checks"][name]
            log(stage="validated", size=size, path=name, **result["checks"][name])
    result["checks"]["atom_dot_local"] = check(
        layer(x, backend="triton_atom_dot"), expected
    )
    assert result["checks"]["atom_dot_local"]["passed"]
    result["extra_peak_bytes"] = {
        name: peak_extra(fn) for name, fn in functions.items()
    }
    save(result)
    result.update(timing(functions))
    save(result)
    log(stage="timed", size=size, medians=result["median_ms"])
    view = SimpleNamespace(out_features=size, atoms=site.atoms)
    result["profiles"] = []
    for name in ("prepare_exhaustive", "prepare_local", "fused_local"):
        result["profiles"].append(capture(view, x, name, functions[name], False, out))
        save(result)
    result["completed"] = True
    save(result)
    log(stage="completed", size=size)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 8192])
    args = parser.parse_args()
    assert len(args.source_commit) == 40 and all(
        c in "0123456789abcdef" for c in args.source_commit
    )
    assert all(n > 0 and n % 64 == 0 for n in args.sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "precision": "FP32 TF32 off",
        "tile": [64, 64],
        "timing": "CUDA Graph, median of 3 rounds rep=20ms, paired old/new on same tensors",
        "cases": [],
    }
    for size in args.sizes:
        out = args.output_dir / str(size)
        out.mkdir(exist_ok=True)
        result["cases"].append({})

        def save(case):
            result["cases"][-1] = case
            (args.output_dir / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

        run(size, out, save)
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
