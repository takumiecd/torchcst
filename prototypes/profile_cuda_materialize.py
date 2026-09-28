"""Compare a CUDA C++ local W kernel with the Triton bounded-list kernel."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    trainable_boxed_prepare,
)
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    rows = min(1024, n // 2)
    source = Path(__file__).with_name("cuda") / "materialize_bounded.cu"
    extension = load(
        name="torchcst_materialize_bounded_probe",
        sources=[str(source)],
        extra_cuda_cflags=["--fmad=false", "-O3"],
        verbose=False,
    )
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists, counts, cap = build_listed_forward_candidates_bounded(
        layer, packed, circle, section, offsets
    )
    reference = torch.empty((rows, n), device="cuda")
    output = torch.empty_like(reference)

    def triton_run():
        materialize_listed[(rows // 64 * layer.column_groups * 4,)](
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            reference,
            K=n,
            CG=layer.column_groups,
            G=layer.strip.chart.tile_count,
            PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
            MAX_CANDIDATES=cap,
            STATION_START=0,
            ROW_START=0,
            BOUNDED=True,
            num_warps=1,
            enable_fp_fusion=False,
        )

    def cuda_run(threads):
        extension.materialize(
            packed,
            circle,
            section,
            lists,
            counts,
            offsets,
            output,
            layer.column_groups,
            layer.strip.chart.tile_count,
            cap,
            0,
            0,
            threads,
        )

    triton_run()
    errors = {}
    graphs = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        triton_run()
    graphs["triton"] = graph
    for threads in (32, 64, 128, 256, 512):
        cuda_run(threads)
        torch.cuda.synchronize()
        difference = (output - reference).abs()
        errors[str(threads)] = {
            "max_abs": difference.max().item(),
            "relative_l2": (
                difference.norm() / reference.norm().clamp_min(1e-30)
            ).item(),
        }
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cuda_run(threads)
        graphs[str(threads)] = graph
    samples = {name: [] for name in graphs}
    names = tuple(graphs)
    for i in range(32):
        order = names[i % len(names) :] + names[: i % len(names)]
        for name in order:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "window_rows": rows,
        "capacity": cap,
        "max_abs_vs_triton": errors,
        "median_ms": {name: statistics.median(v) for name, v in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
