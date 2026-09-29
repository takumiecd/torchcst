"""Check whether a full native CST AdamW step can be captured after CSR lists."""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from prototypes.block_streamed_backward import mapped_streamed_trainable
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--rows", type=int, choices=(16, 128, 2048), required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--window-rows", type=int, default=None)
    parser.add_argument("--cache-windows", type=int, default=None)
    parser.add_argument("--verify-steps", type=int, default=0)
    parser.add_argument("--listed-unroll", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--listed-builder-ba", type=int, default=8)
    parser.add_argument("--listed-builder-warps", type=int, default=4)
    parser.add_argument(
        "--optimizer-mode", choices=("foreach", "fused"), default="foreach"
    )
    parser.add_argument(
        "--materialize-mode",
        choices=("listed", "listed_csr", "listed_bounded"),
        default="listed_csr",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(layer.strip)
    x = torch.randn((m, n), device="cuda", requires_grad=True)
    dy = torch.randn((m, n), device="cuda")
    optimizer_kwargs = {args.optimizer_mode: True, "capturable": True}
    optimizer = torch.optim.AdamW([layer.strip.atoms.p], lr=1e-3, **optimizer_kwargs)
    mode = "ieee" if m <= 128 else "fp16x3_dx"
    forward_mode = "ieee" if m <= 128 else "fp16x3"
    window_rows = min(1024, n // 2) if args.window_rows is None else args.window_rows
    cache_windows = (
        (0 if n == 1024 else 2) if args.cache_windows is None else args.cache_windows
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        y = mapped_streamed_trainable(
            layer,
            x,
            boxes=boxes,
            witness_cols=hints,
            atom_kernel="staged_listed",
            window_rows=window_rows,
            cache_windows=cache_windows,
            gemm_mode=mode,
            forward_gemm_mode=forward_mode,
            materialize_mode=args.materialize_mode,
            listed_unroll=args.listed_unroll,
            listed_builder_ba=args.listed_builder_ba,
            listed_builder_warps=args.listed_builder_warps,
        )
        y.backward(dy)
        optimizer.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    result = {
        "shape": [m, n, n],
        "device": torch.cuda.get_device_name(),
        "materialize_mode": args.materialize_mode,
        "listed_unroll": args.listed_unroll,
        "listed_builder_ba": args.listed_builder_ba,
        "listed_builder_warps": args.listed_builder_warps,
        "optimizer_mode": args.optimizer_mode,
        "cache_windows": cache_windows,
        "window_rows": window_rows,
        "allocated_after_warmup": torch.cuda.memory_allocated(),
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.graph(graph):
            step()
        torch.cuda.synchronize()
        result["capture_peak_allocated"] = torch.cuda.max_memory_allocated()
        result["allocated_after_capture"] = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        before = layer.strip.atoms.p.detach().clone()
        graph.replay()
        torch.cuda.synchronize()
        after = layer.strip.atoms.p.detach().clone()
        result["first_replay_peak_allocated"] = torch.cuda.max_memory_allocated()
        result["capture_success"] = True
        result["parameters_changed"] = not torch.equal(before, after)
        samples = []
        for _ in range(args.rounds):
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        result["median_graph_replay_ms"] = statistics.median(samples)
        result["samples_ms"] = samples
        result["steady_replay_peak_allocated"] = torch.cuda.max_memory_allocated()
        if args.verify_steps:
            eager_layer = copy.deepcopy(layer)
            eager_x = x.detach().clone().requires_grad_()
            eager_optimizer = torch.optim.AdamW(
                [eager_layer.strip.atoms.p], lr=1e-3, **optimizer_kwargs
            )
            eager_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            deviations = []
            for _ in range(args.verify_steps):
                graph.replay()
                eager_optimizer.zero_grad(set_to_none=True)
                eager_x.grad = None
                eager_y = mapped_streamed_trainable(
                    eager_layer,
                    eager_x,
                    boxes=boxes,
                    witness_cols=hints,
                    atom_kernel="staged_listed",
                    window_rows=window_rows,
                    cache_windows=cache_windows,
                    gemm_mode=mode,
                    forward_gemm_mode=forward_mode,
                    materialize_mode=args.materialize_mode,
                    listed_unroll=args.listed_unroll,
                    listed_builder_ba=args.listed_builder_ba,
                    listed_builder_warps=args.listed_builder_warps,
                )
                eager_y.backward(dy)
                eager_optimizer.step()
                torch.cuda.synchronize()
                graph_p = layer.strip.atoms.p.detach()
                eager_p = eager_layer.strip.atoms.p.detach()
                difference = graph_p - eager_p
                deviations.append(
                    {
                        "max_abs": difference.abs().max().item(),
                        "relative_l2": (
                            difference.norm() / eager_p.norm().clamp_min(1e-30)
                        ).item(),
                    }
                )
            result["parameter_deviations_vs_eager"] = deviations
    except RuntimeError as exc:
        result["capture_success"] = False
        result["error_class"] = type(exc).__name__
        result["error"] = str(exc).splitlines()[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
