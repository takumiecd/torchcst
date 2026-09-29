"""Compare listed, atom-parallel, and unlisted 1024² W generation."""

import argparse
import json
from pathlib import Path
import torch
import triton
from triton.testing import do_bench_cudagraph
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.block_streamed_backward import (
    trainable_boxed_prepare,
    build_listed_forward_candidates_bounded,
)
from prototypes.block_materialize_listed import materialize_listed
from prototypes.block_materialize_parallel import materialize_listed_parallel
from prototypes.block_materialize_kernel import materialize_logical
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import execution_plan, PROFILE_KINDS

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(21)
n = 1024
layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
plan = execution_plan(layer.strip)
boxes = station_site_boxes(plan.circle, plan.section, 64)
hints = balanced_home_columns(layer.strip)
p, circle, section, offsets = trainable_boxed_prepare(
    layer.strip, layer.strip.atoms.p, boxes=boxes, witness_cols=hints
)
lists, counts, cap = build_listed_forward_candidates_bounded(
    layer, p, circle, section, offsets, ba=32, warps=1
)
g = layer.strip.chart.tile_count
cg = layer.column_groups
profile = PROFILE_KINDS[type(layer.strip.kernel.profile)]
w = torch.empty((512, n), device="cuda")


def listed():
    materialize_listed[(512 // 64 * cg * 4,)](
        p,
        circle,
        section,
        lists,
        counts,
        offsets,
        w,
        K=n,
        CG=cg,
        G=g,
        PROFILE=profile,
        MAX_CANDIDATES=cap,
        BOUNDED=True,
        BR=16,
        LOOP_UNROLL=4,
        STATION_START=0,
        ROW_START=0,
        num_warps=1,
        enable_fp_fusion=False,
    )


def parallel(ba, warps):
    materialize_listed_parallel[(512 // 64 * cg * 4,)](
        p,
        circle,
        section,
        lists,
        counts,
        offsets,
        w,
        K=n,
        CG=cg,
        G=g,
        PROFILE=profile,
        MAX_CANDIDATES=cap,
        BOUNDED=True,
        BA=ba,
        STATION_START=0,
        ROW_START=0,
        num_warps=warps,
        enable_fp_fusion=False,
    )


def direct(bn, bk, ba):
    materialize_logical[(triton.cdiv(512, bn), cg, 64 // bk)](
        p,
        circle,
        section,
        offsets,
        w,
        N=n,
        K=n,
        S=64,
        T=64,
        CG=cg,
        G=g,
        D=4,
        PROFILE=profile,
        BN=bn,
        BK=bk,
        BA=ba,
        FACTORED=True,
        ROW_GROUP_START=0,
        LOCAL_W=True,
        num_warps=4,
        enable_fp_fusion=False,
    )


listed()
ref = w.clone()
results = []
for name, fn in (
    [("listed", listed)]
    + [
        (f"parallel-ba{ba}-w{warps}", lambda ba=ba, warps=warps: parallel(ba, warps))
        for ba, warps in [(2, 4), (2, 8), (4, 4), (4, 8)]
    ]
    + [
        (f"direct-{bn}-{bk}-{ba}", lambda bn=bn, bk=bk, ba=ba: direct(bn, bk, ba))
        for bn, bk, ba in [(16, 32, 1), (32, 32, 1), (64, 32, 1), (16, 64, 1)]
    ]
):
    try:
        fn()
        torch.cuda.synchronize()
        delta = float((w - ref).abs().max())
        ms = do_bench_cudagraph(fn, rep=100)
        results.append({"algorithm": name, "us": ms * 1000, "max_delta": delta})
    except Exception as e:
        results.append({"algorithm": name, "error": str(e)[:500]})
    print(results[-1], flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps({"device": torch.cuda.get_device_name(), "results": results}, indent=2)
    + "\n"
)
