"""Screen exact listed on-chip CST forward against materialize-then-GEMM."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton as tr

from prototypes.benchmark_tile_study import mapped_control
from prototypes.block_streamed_backward import (
    build_listed_forward_candidates_bounded,
    trainable_boxed_prepare,
)
from prototypes.block_streamed_forward import streamed_forward
from prototypes.block_strip_linear import BlockStripLinear
from prototypes.flash_listed_forward import (
    flash_listed_forward,
    reduce_flash_partials,
)
from prototypes.support_box_routing import balanced_home_columns, station_site_boxes
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan, prepare


def metrics(actual, reference):
    delta = actual - reference
    return {
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
        "violations": int((delta.abs() > 3e-5 + 3e-5 * reference.abs()).sum()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(16, 128), required=True)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    n = 1024
    layer = BlockStripLinear((n, n), (64, 64), round(0.05 * n * n), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    lists, counts, capacity = build_listed_forward_candidates_bounded(
        layer, packed, circle, section, offsets, max_candidates=192, ba=32, warps=1
    )
    small_lists, small_counts, small_capacity = build_listed_forward_candidates_bounded(
        layer,
        packed,
        circle,
        section,
        offsets,
        max_candidates=192,
        ba=32,
        warps=1,
        br=8,
    )
    weight, canonical = mapped_control(
        layer,
        prepare(layer.strip, layer.strip.atoms.p, support_layout=True),
        canonical_chunk=4,
    )
    assert canonical["passed"], canonical
    x = torch.randn(args.rows, n, device="cuda")
    reference = x @ weight.T
    del weight
    outputs = {}
    resources = {}
    runners = {}

    def baseline(tile_rows, listed_data, name):
        outputs[name] = streamed_forward(
            layer,
            x,
            prepared=(packed, circle, section, offsets),
            weight_chunk_rows=512,
            cache_weight_rows=0,
            gemm_mode="ieee",
            materialize_mode="listed_bounded",
            listed_data=listed_data,
            listed_tile_shape=(tile_rows, 64),
            listed_unroll=4,
        )

    runners["materialize_gemm_16"] = lambda: baseline(
        16, (lists, counts, capacity), "materialize_gemm_16"
    )
    runners["materialize_gemm_8"] = lambda: baseline(
        8, (small_lists, small_counts, small_capacity), "materialize_gemm_8"
    )
    variants = [(16, chunks) for chunks in (1, 2, 4)]
    if args.rows == 128:
        variants.extend((bm, 1) for bm in (32, 64, 128))
    for bm, chunks in variants:
        splits = 16 // chunks
        partial = torch.empty((splits, args.rows, n), device="cuda")
        y = torch.empty_like(reference)
        name = f"flash_bm{bm}_chunks_{chunks}"

        def run(bm=bm, chunks=chunks, splits=splits, partial=partial, y=y, name=name):
            compiled = flash_listed_forward[(n // 16, splits, tr.cdiv(args.rows, bm))](
                x,
                packed,
                circle,
                section,
                offsets,
                lists,
                counts,
                partial,
                M=args.rows,
                N=n,
                K=n,
                CG=layer.column_groups,
                G=layer.strip.chart.tile_count,
                PROFILE=PROFILE_KINDS[type(layer.strip.kernel.profile)],
                MAX_CANDIDATES=capacity,
                CHUNKS_PER_PROGRAM=chunks,
                BM=bm,
                num_warps=4,
                enable_fp_fusion=False,
            )
            reduce_flash_partials[(tr.cdiv(args.rows, bm), n // 16)](
                partial,
                y,
                M=args.rows,
                N=n,
                SPLITS=splits,
                BM=bm,
                num_warps=4,
            )
            outputs[name] = y
            resources[name] = {
                "registers_per_thread": compiled.n_regs,
                "spills": compiled.n_spills,
                "partial_bytes": partial.numel() * partial.element_size(),
            }

        runners[name] = run
    for runner in runners.values():
        runner()
    torch.cuda.synchronize()
    checks = {name: metrics(output, reference) for name, output in outputs.items()}
    for name, check in checks.items():
        if check["violations"]:
            raise AssertionError(f"{name}: {check}")
    samples = {name: [] for name in runners}
    if args.rounds:
        graphs = {}
        for name, runner in runners.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner()
            graphs[name] = graph
        names = tuple(graphs)
        for round_index in range(args.rounds):
            order = (
                names[round_index % len(names) :] + names[: round_index % len(names)]
            )
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                graphs[name].replay()
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": tr.__version__,
        "rows": args.rows,
        "rounds": args.rounds,
        "candidate_capacity": capacity,
        "candidate_overflows": int((counts < 0).sum()),
        "canonical": canonical,
        "scope": "prepared candidates; exact CST forward; CUDA Graph replay; same process",
        "checks": checks,
        "resources": resources,
        "median_ms": {
            name: statistics.median(values) if values else None
            for name, values in samples.items()
        },
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
