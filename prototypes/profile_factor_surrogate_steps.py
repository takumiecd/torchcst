"""Compare a trainable rank-128 surrogate with dense and atom-CST steps."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    weight, canonical = mapped_control(
        layer,
        prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
        canonical_chunk=4,
    )
    assert canonical["passed"], canonical
    with torch.no_grad():
        left, singular, right_t = torch.linalg.svd(weight, full_matrices=False)
        scale = singular[: args.rank].sqrt()
        left_factor = torch.nn.Parameter((left[:, : args.rank] * scale).contiguous())
        right_factor = torch.nn.Parameter(
            (right_t[: args.rank, :].T * scale).contiguous()
        )
        dense_weight = torch.nn.Parameter(weight.detach().clone())
        x = torch.randn(args.rows, n, device="cuda", requires_grad=True)
        dy = torch.randn(args.rows, n, device="cuda")
        target = F.linear(x, dense_weight)
        factor_output = (x @ right_factor) @ left_factor.T
        factor_error = {
            "relative_l2": float(
                (factor_output - target).norm() / target.norm().clamp_min(1e-30)
            ),
            "max_abs": float((factor_output - target).abs().max()),
        }
        del target, factor_output, left, singular, right_t, weight
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    optimizers = {
        "dense": torch.optim.AdamW(
            [dense_weight], lr=1e-3, foreach=True, capturable=True
        ),
        "factor": torch.optim.AdamW(
            [left_factor, right_factor], lr=1e-3, foreach=True, capturable=True
        ),
        "cst": torch.optim.AdamW(
            [layer.strip.atoms.p], lr=1e-3, foreach=True, capturable=True
        ),
    }

    def step(mode):
        optimizers[mode].zero_grad(set_to_none=True)
        x.grad = None
        if mode == "dense":
            output = F.linear(x, dense_weight)
        elif mode == "factor":
            output = (x @ right_factor) @ left_factor.T
        else:
            output = mapped_streamed_trainable(
                layer,
                x,
                boxes=boxes,
                witness_cols=hints,
                window_rows=512,
                cache_windows=2,
                cache_weight_dtype=torch.float16,
                weight_tile_rows=8,
                atom_kernel="staged_listed",
                gemm_mode="ieee",
                forward_gemm_mode="ieee",
                materialize_mode="listed_bounded",
                listed_unroll=4,
                listed_builder_ba=32,
                listed_builder_warps=1,
            )
        output.backward(dy)
        optimizers[mode].step()

    modes = ("dense", "factor", "cst")
    for mode in (*modes, *reversed(modes)):
        step(mode)
    torch.cuda.synchronize()
    graphs = {}
    for mode in modes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step(mode)
        graphs[mode] = graph
    torch.cuda.synchronize()
    samples = {mode: [] for mode in modes}
    for index in range(args.rounds):
        order = modes[index % len(modes) :] + modes[: index % len(modes)]
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[mode].replay()
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "rows": args.rows,
        "rank": args.rank,
        "factor_parameter_count": left_factor.numel() + right_factor.numel(),
        "atom_parameter_count": layer.strip.atoms.p.numel(),
        "factor_initial_output_error": factor_error,
        "scope": "complete CUDA Graph AdamW steps; SVD initialization outside timing; all modes coexist",
        "cases": {
            mode: {
                "median_ms": statistics.median(samples[mode]),
                "samples_ms": samples[mode],
            }
            for mode in modes
        },
        "paired_factor_vs_cst": statistics.median(
            factor / cst for factor, cst in zip(samples["factor"], samples["cst"])
        ),
        "paired_factor_vs_dense": statistics.median(
            factor / dense for factor, dense in zip(samples["factor"], samples["dense"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
