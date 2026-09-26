"""Bounded search of local accumulation and input reuse on target-density A100."""

import argparse
import gc
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton

from prototypes.benchmark_large_forward import check, peak_extra
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_fused_config import FusedConfig, launch_fused
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.profile_current_paths import capture, resource_record
from torchcst.nn._backends._preparation import prepare

CONFIGS = [
    "64,16,16,8,4,0",
    "128,16,16,8,4,0",
    "64,16,16,32,4,0",
    "128,16,16,32,4,0",
    "64,16,16,8,4,1",
    "128,16,16,8,4,1",
    "64,16,16,32,4,1",
    "128,16,16,32,4,1",
    "128,16,32,16,4,0",
    "128,16,32,16,4,1",
    "128,32,16,16,4,0",
    "128,32,16,16,4,1",
    "128,16,64,8,4,1",
    "128,16,64,8,8,1",
]


def log(**value):
    print(json.dumps(value), flush=True)


def config(text):
    values = list(map(int, text.split(",")))
    assert len(values) in (6, 7, 8) and all(value in (0, 1) for value in values[5:])
    return FusedConfig(*values[:5], *(bool(value) for value in values[5:]))


@torch.no_grad()
def run(size, batch, atoms, configs, full, out, save):
    torch.manual_seed(21)
    result = {
        "size": size,
        "batch": batch,
        "atoms": atoms,
        "full": full,
        "configs": {k: asdict(v) for k, v in configs.items()},
    }
    save(result)
    log(stage="start", size=size, batch=batch, atoms=atoms)
    layer = BlockStripLinear((size, size), (64, 64), atoms, device="cuda")
    prepared = prepare(layer.strip, layer.strip.atoms.p, support_layout=True)
    weight, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    result["canonical"] = canonical
    assert canonical["passed"]
    x = torch.randn(batch, size, device="cuda")
    expected = F.linear(x, weight)
    functions = {"dense": partial(F.linear, x, weight)}
    result["checks"] = {}
    result["resources"] = {}
    result["errors"] = {}
    for name, cfg in configs.items():
        fn = partial(
            layer,
            x,
            backend="triton_fused",
            fused_config=cfg,
            prepared=None if full else prepared,
        )
        try:
            log(stage="compile", config=name)
            validation = check(fn(), expected)
            result["checks"][name] = validation
            y = torch.empty_like(expected)
            compiled = launch_fused(layer, x, prepared, y, cfg)
            record = resource_record(
                compiled,
                [
                    (batch + cfg.batch_rows - 1) // cfg.batch_rows,
                    layer.row_groups * ((64 + cfg.output_rows - 1) // cfg.output_rows),
                ],
            )
            record["threads_per_block"] = 32 * cfg.warps
            result["resources"][name] = record
            del y
            if validation["passed"]:
                functions[name] = fn
            log(stage="validated", config=name, **validation, resources=record)
        except (RuntimeError, triton.OutOfResources) as error:
            result["errors"][name] = str(error)
            log(stage="config_error", config=name, error=str(error))
        save(result)
    result["extra_peak_bytes"] = {
        name: peak_extra(fn) for name, fn in functions.items()
    }
    result.update(timing(functions))
    save(result)
    fastest = min(
        (n for n in functions if n != "dense"), key=lambda n: result["median_ms"][n]
    )
    result["fastest"] = fastest
    view = SimpleNamespace(out_features=size, atoms=layer.strip.atoms)
    baseline = next(iter(configs))
    result["profiles"] = [
        capture(view, x, name.replace(",", "_"), functions[name], False, out)
        for name in dict.fromkeys((baseline, fastest))
        if name in functions
    ]
    result["completed"] = True
    save(result)
    log(
        stage="completed",
        size=size,
        batch=batch,
        fastest=fastest,
        medians=result["median_ms"],
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cases", nargs="+", default=["4096:128:838861"])
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    assert len(args.source_commit) == 40 and all(
        c in "0123456789abcdef" for c in args.source_commit
    )
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
        "timing": "CUDA Graph 3 rounds rep=20ms; full flag controls preparation inclusion",
        "cases": [],
    }
    configs = {spec: config(spec) for spec in args.configs}
    for spec in args.cases:
        size, batch, atoms = map(int, spec.split(":"))
        assert size % 64 == 0 and min(size, batch, atoms) > 0
        out = args.output_dir / spec.replace(":", "_")
        out.mkdir(exist_ok=True)
        result["cases"].append({})

        def save(case):
            result["cases"][-1] = case
            (args.output_dir / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )

        run(size, batch, atoms, configs, args.full, out, save)
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
