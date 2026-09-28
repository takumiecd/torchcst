"""Measure when refreshing a mapped 5% weight once beats streaming each input.

All paths use the same atom values and logical weight. A refresh is required
after every atom update; its cost is included once per group of microbatches.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    balanced_home_columns,
    boxed_prepare,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if not args.microbatches or any(m < 1 for m in args.microbatches):
        parser.error("microbatches must be positive")

    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    inputs = [torch.randn(128, n, device="cuda") for _ in range(max(args.microbatches))]
    cached_w = torch.empty((n, n), device="cuda")

    def prepare_current():
        return boxed_prepare(
            site, site.atoms.p, boxes=boxes, witness_cols=hints, fast_witness=True
        )

    def refresh(packed):
        p, circle, section, offsets = packed
        materialize_logical[(n // 64, layer.column_groups, 2)](
            p,
            circle,
            section,
            offsets,
            cached_w,
            N=n,
            K=n,
            S=64,
            T=64,
            CG=layer.column_groups,
            G=site.chart.tile_count,
            D=4,
            PROFILE=PROFILE_KINDS[type(site.kernel.profile)],
            BN=64,
            BK=32,
            BA=1,
            FACTORED=True,
            num_warps=4,
            enable_fp_fusion=True,
        )

    prepared = prepare_current()
    expected_w, canonical = mapped_control(layer, prepared, canonical_chunk=4)
    assert canonical["passed"]
    refresh(prepared)
    weight_check = check(cached_w, expected_w)
    assert weight_check["passed"], weight_check
    outputs = [F.linear(x, expected_w) for x in inputs]
    checks = {}
    functions = {}
    for count in args.microbatches:

        def streamed_every_time(count=count):
            return [
                layer(x, backend="triton_streamed", prepared=prepare_current())
                for x in inputs[:count]
            ]

        def streamed_prepare_once(count=count):
            current = prepare_current()
            return [
                layer(x, backend="triton_streamed", prepared=current)
                for x in inputs[:count]
            ]

        def refreshed_dense(count=count):
            refresh(prepare_current())
            return [F.linear(x, cached_w) for x in inputs[:count]]

        def stored_dense(count=count):
            return [F.linear(x, expected_w) for x in inputs[:count]]

        for label, fn in (
            ("stream_each", streamed_every_time),
            ("stream_prepare_once", streamed_prepare_once),
            ("refresh_once", refreshed_dense),
            ("stored_dense", stored_dense),
        ):
            name = f"{label}_{count}"
            functions[name] = fn
            actual = fn()
            checks[name] = [check(a, b) for a, b in zip(actual, outputs)]
            assert all(value["passed"] for value in checks[name]), checks[name]

    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [128, n, n],
        "atoms": site.atoms.p.shape[0],
        "canonical": canonical,
        "weight_check": weight_check,
        "checks": checks,
        "cached_weight_bytes": cached_w.numel() * cached_w.element_size(),
        "streamed_weight_window_bytes": min(1024, n) * n * 4,
        "fixed_box_bytes": boxes.numel() * boxes.element_size(),
        "fixed_hint_bytes": hints.numel() * hints.element_size(),
        "timing": "CUDA Graph 3 rounds rep=20ms. Inputs share fixed atom values within each group; refresh_once includes one current preparation and full weight generation. Static geometry and hints are built outside timing.",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
