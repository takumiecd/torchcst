"""Same-run A100 comparison of exact, boxed, and positive-witness support."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    boxed_prepare,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import execution_plan, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    n = args.size
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    x = torch.randn(128, n, device="cuda")

    def boxed():
        return boxed_prepare(site, site.atoms.p, boxes=boxes)

    def witnessed():
        return boxed_prepare(site, site.atoms.p, boxes=boxes, witness_cols=hints)

    def fast_witnessed():
        return boxed_prepare(
            site, site.atoms.p, boxes=boxes, witness_cols=hints, fast_witness=True
        )

    ordinary_layout = prepare(site, site.atoms.p, support_layout=True)
    boxed_layout = boxed()
    witness_layout = witnessed()
    fast_witness_layout = fast_witnessed()
    exact_layout = all(
        torch.equal(ordinary_layout[index], variant[index])
        for variant in (boxed_layout, witness_layout, fast_witness_layout)
        for index in (0, 3)
    )
    assert exact_layout
    expected_w, canonical = mapped_control(layer, ordinary_layout, canonical_chunk=4)
    assert canonical["passed"]
    expected = F.linear(x, expected_w)

    def ordinary_full():
        return layer(x, backend="triton_streamed")

    def boxed_full():
        return layer(x, backend="triton_streamed", prepared=boxed())

    def witnessed_full():
        return layer(x, backend="triton_streamed", prepared=witnessed())

    def fast_witnessed_full():
        return layer(x, backend="triton_streamed", prepared=fast_witnessed())

    functions = {
        "dense_precomputed": partial(F.linear, x, expected_w),
        "fused_default": partial(layer, x, backend="triton_fused"),
        "ordinary_prepare": lambda: prepare(site, site.atoms.p, support_layout=True),
        "boxed_prepare": boxed,
        "witnessed_prepare": witnessed,
        "fast_witnessed_prepare": fast_witnessed,
        "ordinary_full": ordinary_full,
        "boxed_full": boxed_full,
        "witnessed_full": witnessed_full,
        "fast_witnessed_full": fast_witnessed_full,
    }
    checks = {
        name: check(functions[name](), expected)
        for name in (
            "dense_precomputed",
            "fused_default",
            "ordinary_full",
            "boxed_full",
            "witnessed_full",
            "fast_witnessed_full",
        )
    }
    assert all(value["passed"] for value in checks.values())
    eager_names = ("fused_default", "ordinary_full", "fast_witnessed_full")
    for name in eager_names:
        for _ in range(3):
            functions[name]()
    torch.cuda.synchronize()
    eager_samples = {name: [] for name in eager_names}
    for round_index in range(3):
        order = eager_names if round_index % 2 == 0 else tuple(reversed(eager_names))
        for name in order:
            start = time.perf_counter()
            for _ in range(10):
                functions[name]()
            torch.cuda.synchronize()
            eager_samples[name].append((time.perf_counter() - start) * 100)
    eager = {
        name: {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }
        for name, samples in eager_samples.items()
    }
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [128, n, n],
        "atoms": site.atoms.p.shape[0],
        "canonical": canonical,
        "exact_prepared_layout": exact_layout,
        "checks": checks,
        "eager_wall_time": eager,
        "static_box_bytes": boxes.numel() * boxes.element_size(),
        "static_witness_bytes": hints.numel() * hints.element_size(),
        "timing": "CUDA Graph 3 rounds rep=20ms; fixed boxes and home-column hints built once",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
