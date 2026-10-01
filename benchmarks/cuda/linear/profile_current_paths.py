"""Profile current I/B fused and direct paths; distinguish facts from counters.

PyTorch/CUPTI provides kernel durations, not bandwidth utilization or stalls.
Compiler resources are reported separately; they are not achieved occupancy.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import record_function

from benchmarks.cuda.linear._profile_trace import kernel_summary
from benchmarks.cuda.linear.benchmark_triton_linear import model
from experiments.cuda.linear.local_atom_kernels import direct_forward
from experiments.cuda.linear.local_atom_linear import forward as direct
from torchcst._backends.cuda.algorithms.strip_torus.fused.executor import (
    forward as fused,
)
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare
from torchcst._backends.cuda.algorithms.strip_torus.fused.kernels import fused_forward
from torchcst.profiling import CSTProfiler


def resources(layer, x):
    with torch.no_grad():
        p, circle, section, offsets = prepare(layer, layer.atoms.p, support_layout=True)
        bounds = torch.stack((section.amin(0), section.amax(0)))
        y = x.new_empty((x.shape[0], circle.shape[0]))
        common = {
            "M": x.shape[0],
            "N": circle.shape[0],
            "K": x.shape[1],
            "D": p.shape[1] - 2,
            "G": layer.chart.tile_count,
            "PROFILE": 1,
        }
        result = {}
        for bm in (16, 64):
            grid = ((x.shape[0] + bm - 1) // bm, layer.chart.tile_count, 1)
            compiled = fused_forward[grid](
                x,
                p,
                circle,
                section,
                offsets,
                y,
                **common,
                STATION_ROWS=16,
                BM=bm,
                BN=16,
                BK=16,
                BA=8,
                SUPPORT_LAYOUT=True,
                num_warps=4,
                enable_fp_fusion=False,
            )
            result[f"fused_bm{bm}"] = resource_record(compiled, grid)
        for bm in (4, 16):
            grid = ((x.shape[0] + bm - 1) // bm, circle.shape[0])
            compiled = direct_forward[grid](
                x,
                p,
                circle,
                section,
                offsets,
                bounds,
                y,
                **common,
                S=16,
                BM=bm,
                BK=128,
                num_warps=4,
                enable_fp_fusion=False,
            )
            result[f"direct_bm{bm}"] = resource_record(compiled, grid)
        return result


def resource_record(compiled, grid):
    return {
        "grid": grid,
        "registers_per_thread": compiled.n_regs,
        "compiler_spills": compiled.n_spills,
        "shared_bytes_per_block": compiled.metadata.shared,
        "threads_per_block": 128,
    }


def capture(layer, x, name, fn, training, output_dir):
    dy = torch.ones((x.shape[0], layer.out_features), device=x.device)

    def step():
        if training:
            with torch.enable_grad():
                y = fn()
                return torch.autograd.grad(y, (x, layer.atoms.p), dy)
        with torch.no_grad():
            return fn()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with CSTProfiler() as prof:
        for _ in range(3):
            with record_function("measured_step"):
                step()
            torch.cuda.synchronize()
    trace_path = output_dir / f"{name}.trace.json"
    prof.export_chrome_trace(str(trace_path))
    phases = [
        {
            "name": e.key,
            "cpu_us_per_step": e.cpu_time_total / 3,
            "device_us_per_step": e.device_time_total / 3,
        }
        for e in prof.key_averages()
        if e.key.startswith("cst.") or e.key == "measured_step"
    ]
    return {
        "name": name,
        "training": training,
        "steps": 3,
        # Ranges overlap and must never be added to kernel durations.
        "annotation_ranges": phases,
        **kernel_summary(trace_path, steps=3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    layer = model(512, rows=256, columns=512)
    x = torch.randn(128, 512, device="cuda", requires_grad=True)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_commit": args.source_commit,
        "shape": [256, 512],
        "batch": 128,
        "atoms": 512,
        "precision": "FP32 IEEE, TF32 off",
        "compiler_resources": resources(layer, x),
        "profiles": [],
    }
    paths = {
        "fused_bm16": lambda: fused(layer, x, layer.atoms.p, batch_tile=16),
        "fused_bm64": lambda: fused(layer, x, layer.atoms.p, batch_tile=64),
        "direct_bm4": lambda: direct(layer, x, layer.atoms.p, batch_tile=4),
        "direct_bm16": lambda: direct(layer, x, layer.atoms.p, batch_tile=16),
    }
    for name, fn in paths.items():
        measured = capture(layer, x, name, fn, False, args.output_dir)
        result["profiles"].append(measured)
        print(
            json.dumps(
                {
                    "name": name,
                    "kernel_us": measured["kernel_us_per_step"],
                    "top": measured["kernels"][:4],
                }
            ),
            flush=True,
        )
    for name in ("fused_bm16", "direct_bm16"):
        measured = capture(
            layer, x, name + "_train", paths[name], True, args.output_dir
        )
        result["profiles"].append(measured)
        print(
            json.dumps(
                {
                    "name": name + "_train",
                    "kernel_us": measured["kernel_us_per_step"],
                    "top": measured["kernels"][:4],
                }
            ),
            flush=True,
        )
    (args.output_dir / "profile.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
