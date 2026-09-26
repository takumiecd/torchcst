"""Paired batched I/B classification and compact keys at five-percent density."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check, peak_extra
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture
from torchcst.nn._backends._preparation import execution_plan, prepare
from torchcst.nn._backends._triton_preparation import route_and_layout


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
    assert execution_plan(site).routing.local_candidates

    def with_mode(mode, fn):
        def dispatch(*args, **kwargs):
            kwargs["batched_support"] = mode != "legacy"
            kwargs["retain_owners"] = mode != "compact"
            return route_and_layout(*args, **kwargs)

        # Benchmark-only dispatch restores the exact previous key dtype,
        # ownership lifetime and classification kernel for the legacy baseline.
        with patch(
            "torchcst.nn._backends._triton_preparation.route_and_layout", dispatch
        ):
            return fn()

    def prep(mode):
        return with_mode(mode, lambda: prepare(site, site.atoms.p, support_layout=True))

    prepared = prep("compact")
    old_prepared = prep("legacy")
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
        "prepare_legacy": lambda: prep("legacy"),
        "prepare_batched": lambda: prep("batched"),
        "prepare_compact": lambda: prep("compact"),
        "fused_legacy": lambda: with_mode(
            "legacy", lambda: layer(x, backend="triton_fused")
        ),
        "fused_compact": lambda: layer(x, backend="triton_fused"),
        "fused_prepared": lambda: layer(x, backend="triton_fused", prepared=prepared),
    }
    result["checks"] = {}
    for name, fn in functions.items():
        if not name.startswith("prepare_"):
            result["checks"][name] = check(fn(), expected)
            assert result["checks"][name]["passed"], result["checks"][name]
            log(stage="validated", size=size, path=name, **result["checks"][name])
    result["checks"]["atom_dot_compact"] = check(
        layer(x, backend="triton_atom_dot"), expected
    )
    assert result["checks"]["atom_dot_compact"]["passed"]
    result["extra_peak_bytes"] = {
        name: peak_extra(fn) for name, fn in functions.items()
    }
    save(result)
    result.update(timing(functions))
    save(result)
    log(stage="timed", size=size, medians=result["median_ms"])
    view = SimpleNamespace(out_features=size, atoms=site.atoms)
    result["profiles"] = []
    for name in (
        "prepare_legacy",
        "prepare_batched",
        "prepare_compact",
        "fused_compact",
    ):
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
