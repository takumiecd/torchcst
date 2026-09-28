"""Test Ada L2 eviction hints while streaming native CST weight windows."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton as tr

from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import weight_fp_fusion_enabled
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.bounded_gemm import bounded_gemm_kernel
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n, m = args.size, args.rows
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip, layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists, counts, max_candidates = build_listed_forward_candidates(
        layer, packed, circle, section, offsets
    )
    x = torch.randn((m, n), device="cuda")
    y = torch.empty_like(x)
    w = torch.empty((1024, n), device="cuda")
    configs = {
        "default": ("", "", ""),
        "x_last_y_first": ("evict_last", "", "evict_first"),
        "x_last_w_last_y_first": ("evict_last", "evict_last", "evict_first"),
        "w_last_y_first": ("", "evict_last", "evict_first"),
    }

    def run(config):
        a_evict, b_evict, c_evict = config
        for start in range(0, n, 1024):
            materialize_listed[(1024 // 64 * layer.column_groups * 4,)](
                packed, circle, section, lists, counts, offsets, w,
                K=n, CG=layer.column_groups, G=layer.strip.chart.tile_count,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=max_candidates,
                STATION_START=start // 64 * layer.column_groups,
                ROW_START=start,
                num_warps=1,
                enable_fp_fusion=weight_fp_fusion_enabled(w.device),
            )
            b = w.T
            c = y[:, start : start + 1024]
            bm, bn, bk = 32, 128, 32
            bounded_gemm_kernel[(tr.cdiv(m, bm), tr.cdiv(1024, bn))](
                x, b, c, m, 1024, n,
                *x.stride(), *b.stride(), *c.stride(),
                bm, bn, bk, False,
                a_evict, b_evict, c_evict,
                num_warps=4,
            )

    run(configs["default"])
    torch.cuda.synchronize()
    reference = y.clone()
    checks = {}
    for name, config in configs.items():
        run(config)
        torch.cuda.synchronize()
        checks[name] = {
            "max_abs": (y - reference).abs().max().item(),
            "same": bool(torch.equal(y, reference)),
        }
    samples = {name: [] for name in configs}
    names = list(configs)
    for repeat in range(8):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            run(configs[name])
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - begin) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "rows": m,
        "cases": {
            name: {
                "median_ms": statistics.median(samples[name]),
                "samples_ms": samples[name],
                "accuracy": checks[name],
            }
            for name in configs
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
