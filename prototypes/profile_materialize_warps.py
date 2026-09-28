"""Screen warp counts for the local listed W generator on one window."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    build_listed_forward_candidates_csr,
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
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    candidates = {
        "csr": build_listed_forward_candidates_csr(
            layer, packed, circle, section, offsets
        ),
        "bounded": build_listed_forward_candidates_bounded(
            layer, packed, circle, section, offsets
        ),
    }
    w = torch.empty((rows, n), device="cuda")
    grid = (rows // 64 * layer.column_groups * 4,)
    graphs = {}
    errors = {}
    reference = None
    for mode, data in candidates.items():
        if mode == "csr":
            lists, counts, bases, _ = data
            cap = 0
        else:
            lists, counts, cap = data
            bases = None
        for warps in (1, 2, 4, 8):
            name = f"{mode}-{warps}"

            def run(
                lists=lists, counts=counts, cap=cap, mode=mode, bases=bases, warps=warps
            ):
                materialize_listed[grid](
                    packed,
                    circle,
                    section,
                    lists,
                    counts,
                    offsets,
                    w,
                    K=n,
                    CG=layer.column_groups,
                    G=layer.strip.chart.tile_count,
                    PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                    MAX_CANDIDATES=cap,
                    STATION_START=0,
                    ROW_START=0,
                    CSR=mode == "csr",
                    BOUNDED=mode == "bounded",
                    Bases=bases,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )

            run()
            torch.cuda.synchronize()
            if reference is None:
                reference = w.clone()
            errors[name] = (w - reference).abs().max().item()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graphs[name] = graph
    samples = {name: [] for name in graphs}
    names = tuple(graphs)
    for i in range(24):
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
        "bounded_capacity": candidates["bounded"][2],
        "max_abs_vs_csr_1warp": errors,
        "median_ms": {name: statistics.median(v) for name, v in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
