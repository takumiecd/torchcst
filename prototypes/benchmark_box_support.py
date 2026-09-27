"""Measure exact support layout after conservative fixed-site-box culling."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from prototypes.benchmark_large_forward import check
from prototypes.benchmark_tile_study import mapped_control, timing
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import boxed_prepare, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    size = args.size
    assert size in (4096, 8192)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name
    layer = BlockStripLinear(
        (size, size), (64, 64), round(size * size * 0.05), device="cuda"
    )
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    x = torch.randn(128, size, device="cuda")

    def prepare_with_boxes():
        return boxed_prepare(site, site.atoms.p, boxes=boxes)

    ordinary = prepare(site, site.atoms.p, support_layout=True)
    boxed = prepare_with_boxes()
    exact_layout = torch.equal(ordinary[3], boxed[3]) and torch.equal(
        ordinary[0], boxed[0]
    )
    assert exact_layout
    expected_w, canonical = mapped_control(layer, ordinary, canonical_chunk=4)
    assert canonical["passed"]
    expected = F.linear(x, expected_w)

    def ordinary_full():
        return layer(x, backend="triton_streamed")

    def boxed_full():
        return layer(x, backend="triton_streamed", prepared=prepare_with_boxes())

    functions = {
        "ordinary_prepare": partial(prepare, site, site.atoms.p, support_layout=True),
        "boxed_prepare": prepare_with_boxes,
        "ordinary_full": ordinary_full,
        "boxed_full": boxed_full,
    }
    checks = {
        "ordinary_full": check(ordinary_full(), expected),
        "boxed_full": check(boxed_full(), expected),
    }
    assert all(item["passed"] for item in checks.values())
    result = {
        "source_commit": args.source_commit,
        "device": prop.name,
        "multiprocessors": prop.multi_processor_count,
        "shape": [128, size, size],
        "atoms": site.atoms.p.shape[0],
        "canonical": canonical,
        "exact_prepared_layout": exact_layout,
        "checks": checks,
        "static_box_bytes": boxes.numel() * boxes.element_size(),
        "timing": "CUDA Graph 3 rounds rep=20ms; boxes built once from fixed geometry",
        **timing(functions),
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "completed", "medians": result["median_ms"]}))


if __name__ == "__main__":
    main()
