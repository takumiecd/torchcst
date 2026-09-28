"""Break down the 5% weight-first prototype without constructing full W."""

import argparse
import json
import math
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

from prototypes.block_materialize_kernel import materialize_logical
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.support_box_routing import (
    atomic_bucket_sort,
    balanced_home_columns,
    boxed_prepare,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--batch", type=int, choices=(128, 2048), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-materialize", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = args.size
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    site = layer.strip
    plan = execution_plan(site)
    boxes = station_site_boxes(plan.circle, plan.section, 64)
    hints = balanced_home_columns(site)
    x = torch.randn(args.batch, n, device="cuda")

    def prepare_atomic():
        original_sort = torch.sort
        torch.sort = lambda keys, **_kwargs: atomic_bucket_sort(
            keys, 2 * plan.routing.starts.numel() + 1
        )
        try:
            return boxed_prepare(
                site,
                site.atoms.p,
                boxes=boxes,
                witness_cols=hints,
                fast_witness=True,
                fast_decode=True,
            )
        finally:
            torch.sort = original_sort

    packed = prepare_atomic()
    p, circle, section, offsets = packed
    chunk = 1024
    w = x.new_empty((chunk, n))
    y = x.new_empty((args.batch, n))

    def materialize(target, *, bn=64, bk=32, ba=1, factored=True):
        materialize_logical[(math.ceil(chunk / bn), layer.column_groups, 64 // bk)](
            p, circle, section, offsets, target,
            N=n, K=n, S=64, T=64, CG=layer.column_groups,
            G=site.chart.tile_count, D=4,
            PROFILE=PROFILE_KINDS[type(site.kernel.profile)],
            BN=bn, BK=bk, BA=ba, FACTORED=factored,
            ROW_GROUP_START=0, LOCAL_W=True,
            num_warps=4, enable_fp_fusion=True,
        )

    def materialize_one():
        materialize(w)

    def mm_one():
        torch.mm(x, w.T, out=y[:, :chunk])

    def streamed():
        return layer(x, backend="triton_streamed", prepared=packed)

    times = {
        "prepare_ms": do_bench_cudagraph(prepare_atomic, rep=20, return_mode="median"),
        "stream_prepared_ms": do_bench_cudagraph(streamed, rep=20, return_mode="median"),
        "materialize_one_window_ms": do_bench_cudagraph(
            materialize_one, rep=20, return_mode="median"
        ),
        "gemm_one_window_ms": do_bench_cudagraph(mm_one, rep=20, return_mode="median"),
    }
    variants = {}
    if args.scan_materialize:
        materialize_one()
        configs = [
            (16, 32, 1, True), (32, 32, 1, True),
            (32, 64, 1, True), (64, 64, 1, True),
            (64, 32, 2, False), (64, 32, 4, False),
            (64, 32, 8, False), (32, 32, 4, False),
            (16, 32, 8, False),
        ]
        for bn, bk, ba, factored in configs:
            candidate = torch.empty_like(w)

            def candidate_fn():
                materialize(candidate, bn=bn, bk=bk, ba=ba, factored=factored)

            candidate_fn()
            error = (candidate - w).abs()
            valid = torch.isclose(candidate, w, atol=3e-5, rtol=3e-5)
            name = f"{'factored' if factored else 'lanes'}_{bn}x{bk}_ba{ba}"
            variants[name] = {
                "passed": bool(valid.all().item()),
                "max_abs": float(error.max().item()),
                "time_ms": do_bench_cudagraph(candidate_fn, rep=20, return_mode="median"),
            }
    result = {
        "source_commit": args.source_commit,
        "device": torch.cuda.get_device_name(),
        "shape": [args.batch, n, n],
        "atoms": site.atoms.p.shape[0],
        "window_rows": chunk,
        "windows": n // chunk,
        **times,
        "variants": variants,
        "completed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
