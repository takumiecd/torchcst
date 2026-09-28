"""Compare bounded CST window schedules for 1024-square training."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def error(actual, reference, atol):
    delta = (actual - reference).abs()
    return {
        "max_abs": delta.max().item(),
        "reference_max_abs": reference.abs().max().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "violations": (delta > atol + atol * reference.abs()).sum().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--materialize-comparison", action="store_true")
    parser.add_argument("--atom-comparison", action="store_true")
    parser.add_argument("--gemm-comparison", action="store_true")
    parser.add_argument("--combined-comparison", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, m = 1024, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn((m, n), device="cuda", requires_grad=True)
    dy = torch.randn((m, n), device="cuda")
    optimizer = torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, foreach=True)
    if args.combined_comparison:
        configs = (
            (512, 0, "listed", "staged_listed", "ieee", "ieee"),
            (512, 0, "listed", "staged_listed", "fp16x3_dx", "fp16x3"),
            (512, 0, "listed_parallel", "staged_listed", "ieee", "ieee"),
            (512, 0, "listed_parallel", "staged_listed", "fp16x3_dx", "fp16x3"),
        )
    elif args.gemm_comparison:
        configs = tuple(
            (512, 0, "listed", "staged_listed", backward, forward)
            for backward, forward in (
                ("ieee", "ieee"),
                ("ieee", "fp16x3"),
                ("fp16x3_dx", "ieee"),
                ("fp16x3_dx", "fp16x3"),
                ("tf32x3_dx", "tf32x3"),
            )
        )
    elif args.atom_comparison:
        configs = tuple(
            (512, 0, "listed", atom, "ieee", "ieee")
            for atom in ("staged_listed", "atom_major", "interval", "staged", "fused")
        )
    elif args.materialize_comparison:
        configs = (
            (512, 0, "listed", "staged_listed", "ieee", "ieee"),
            (512, 0, "listed_parallel", "staged_listed", "ieee", "ieee"),
        )
    else:
        configs = tuple(
            (window, cache, "listed", "staged_listed", "ieee", "ieee")
            for window, cache in ((512, 0), (256, 1), (256, 2), (128, 4))
        )
    assert all((window * (cache + 1)) < n for window, cache, *_ in configs)

    def forward(config):
        window, cache, mode, atom, backward_gemm, forward_gemm = config
        return mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            window_rows=window,
            cache_windows=cache,
            atom_kernel=atom,
            gemm_mode=backward_gemm,
            forward_gemm_mode=forward_gemm,
            materialize_mode=mode,
        )

    def gradients(config):
        y = forward(config)
        dx, dp = torch.autograd.grad(y, (x, layer.strip.atoms.p), dy)
        return y.detach(), dx.detach(), dp.detach()

    reference = gradients(configs[0])
    checks = {}
    for config in configs:
        actual = gradients(config)
        checks[str(config)] = {
            "output": error(actual[0], reference[0], 3e-5),
            "dx": error(actual[1], reference[1], 3e-5),
            "dp": error(actual[2], reference[2], 3e-4),
        }
        if (
            checks[str(config)]["output"]["violations"]
            or checks[str(config)]["dx"]["violations"]
            or checks[str(config)]["dp"]["relative_l2"] > 1e-5
        ):
            print(
                json.dumps({"failed_config": config, "checks": checks[str(config)]}),
                flush=True,
            )
            raise AssertionError("window schedule changed numerical results")
    del reference, actual

    def step(config):
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = forward(config)
        y.backward(dy)
        optimizer.step()

    for config in configs:
        step(config)
    torch.cuda.synchronize()
    samples = {str(config): [] for config in configs}
    peaks = {str(config): [] for config in configs}
    for round_index in range(args.rounds):
        order = (
            configs[round_index % len(configs) :]
            + configs[: round_index % len(configs)]
        )
        for config in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            step(config)
            torch.cuda.synchronize()
            name = str(config)
            samples[name].append((time.perf_counter() - start) * 1000)
            peaks[name].append(torch.cuda.max_memory_allocated())
    result = {
        "device": torch.cuda.get_device_name(),
        "shape": [m, n, n],
        "scope": "full AdamW step, alternating window schedules on one updating model; all live W windows below full W",
        "checks": checks,
        "cases": {
            str(config): {
                "median_ms": statistics.median(samples[str(config)]),
                "samples_ms": samples[str(config)],
                "peak_bytes_max": max(peaks[str(config)]),
            }
            for config in configs
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
