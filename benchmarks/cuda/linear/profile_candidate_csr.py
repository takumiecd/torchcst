"""Compare synchronization-free atom-list reservations with the dense tile list."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.cuda.linear.block_materialize_listed import materialize_listed
from experiments.cuda.linear.block_streamed_backward import (
    build_listed_forward_candidates,
    build_listed_forward_candidates_bounded,
    build_listed_forward_candidates_csr,
    trainable_boxed_prepare,
)
from experiments.cuda.linear.block_strip_linear import BlockStripLinear
from experiments.cuda.linear.block_tile_atom_lists import mapped_backward_atoms_listed
from experiments.cuda.linear.support_box_routing import (
    balanced_home_columns,
    station_site_boxes,
)
from torchcst.nn._backends._preparation import PROFILE_KINDS, execution_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(1024, 8192), required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(21)
    n = args.size
    chunk = min(1024, n // 2)
    layer = BlockStripLinear((n, n), (64, 64), round(n * n * 0.05), device="cuda")
    plan = execution_plan(layer.strip)
    packed, circle, section, offsets = trainable_boxed_prepare(
        layer.strip,
        layer.strip.atoms.p,
        boxes=station_site_boxes(plan.circle, plan.section, 64),
        witness_cols=balanced_home_columns(layer.strip),
    )
    original = build_listed_forward_candidates(layer, packed, circle, section, offsets)
    csr = build_listed_forward_candidates_csr(layer, packed, circle, section, offsets)
    bounded = build_listed_forward_candidates_bounded(
        layer, packed, circle, section, offsets
    )
    lists, counts, max_candidates = original
    csr_lists, csr_counts, bases, cursor = csr
    bounded_lists, bounded_counts, bounded_capacity = bounded
    assert torch.equal(counts, csr_counts)
    positions = torch.arange(max_candidates, device="cuda")[None, None, :]
    valid = positions < counts[..., None]
    csr_positions = torch.where(valid, bases[..., None] + positions, 0)
    assert torch.equal(lists[valid].to(torch.int32), csr_lists[csr_positions][valid])
    assert cursor.item() <= csr_lists.numel()
    assert torch.equal(counts, bounded_counts)
    bounded_positions = torch.arange(bounded_capacity, device="cuda")[None, None, :]
    bounded_valid = bounded_positions < counts[..., None]
    assert torch.equal(
        lists[:, :, :bounded_capacity][bounded_valid].to(torch.int32),
        bounded_lists[bounded_valid].to(torch.int32),
    )

    common = {
        "K": n,
        "CG": layer.column_groups,
        "G": layer.strip.chart.tile_count,
        "PROFILE": PROFILE_KINDS[type(layer.strip.kernel.profile)],
        "STATION_START": 0,
        "ROW_START": 0,
        "enable_fp_fusion": True,
    }
    w = torch.empty((chunk, n), device="cuda")
    reference_w = torch.empty_like(w)
    grid = (chunk // 64 * layer.column_groups * 4,)
    materialize_listed[grid](
        packed,
        circle,
        section,
        lists,
        counts,
        offsets,
        reference_w,
        MAX_CANDIDATES=max_candidates,
        num_warps=1,
        **common,
    )
    materialize_listed[grid](
        packed,
        circle,
        section,
        csr_lists,
        csr_counts,
        offsets,
        w,
        MAX_CANDIDATES=0,
        CSR=True,
        Bases=bases,
        num_warps=1,
        **common,
    )
    weight_max_abs = (w - reference_w).abs().max().item()
    bounded_w = torch.empty_like(w)
    materialize_listed[grid](
        packed,
        circle,
        section,
        bounded_lists,
        bounded_counts,
        offsets,
        bounded_w,
        MAX_CANDIDATES=bounded_capacity,
        BOUNDED=True,
        num_warps=1,
        **common,
    )
    bounded_weight_max_abs = (bounded_w - reference_w).abs().max().item()
    dw = torch.randn_like(w)
    dp = torch.zeros_like(packed)
    reference_dp = torch.zeros_like(packed)
    backward = {
        "BA": 1,
        "COMPACT": True,
        "BR": 16,
        "BC": 64,
        "OPT_TRIWEIGHT": True,
    }
    mapped_backward_atoms_listed[(grid[0], 1)](
        dw,
        packed,
        circle,
        section,
        lists,
        counts,
        offsets,
        reference_dp,
        MAX_CANDIDATES=max_candidates,
        num_warps=1,
        **backward,
        **common,
    )
    mapped_backward_atoms_listed[(grid[0], 1)](
        dw,
        packed,
        circle,
        section,
        csr_lists,
        csr_counts,
        offsets,
        dp,
        MAX_CANDIDATES=0,
        CSR=True,
        Bases=bases,
        num_warps=1,
        **backward,
        **common,
    )
    gradient = dp - reference_dp
    gradient_max_abs = gradient.abs().max().item()
    gradient_relative_l2 = (
        gradient.norm() / reference_dp.norm().clamp_min(1e-30)
    ).item()
    bounded_dp = torch.zeros_like(packed)
    mapped_backward_atoms_listed[(grid[0], 1)](
        dw,
        packed,
        circle,
        section,
        bounded_lists,
        bounded_counts,
        offsets,
        bounded_dp,
        MAX_CANDIDATES=bounded_capacity,
        BOUNDED=True,
        num_warps=1,
        **backward,
        **common,
    )
    bounded_gradient = bounded_dp - reference_dp
    bounded_gradient_max_abs = bounded_gradient.abs().max().item()
    bounded_gradient_relative_l2 = (
        bounded_gradient.norm() / reference_dp.norm().clamp_min(1e-30)
    ).item()
    torch.cuda.synchronize()

    samples = {"dynamic": [], "fixed": [], "csr": [], "bounded": []}
    for i in range(args.rounds):
        order = ("dynamic", "fixed", "csr", "bounded")
        order = order[i % 4 :] + order[: i % 4]
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if mode == "csr":
                build_listed_forward_candidates_csr(
                    layer, packed, circle, section, offsets
                )
            elif mode == "bounded":
                build_listed_forward_candidates_bounded(
                    layer, packed, circle, section, offsets
                )
            else:
                build_listed_forward_candidates(
                    layer,
                    packed,
                    circle,
                    section,
                    offsets,
                    unchecked_fixed_capacity=max_candidates
                    if mode == "fixed"
                    else None,
                )
            torch.cuda.synchronize()
            samples[mode].append((time.perf_counter() - start) * 1000)
    result = {
        "device": torch.cuda.get_device_name(),
        "size": n,
        "capacity": max_candidates,
        "csr_reserved_entries": cursor.item(),
        "csr_allocated_entries": csr_lists.numel(),
        "bounded_capacity": bounded_capacity,
        "bounded_dtype": str(bounded_lists.dtype),
        "bounded_allocated_bytes": bounded_lists.numel() * bounded_lists.element_size(),
        "checks": {
            "exact_lists": True,
            "weight_max_abs": weight_max_abs,
            "bounded_weight_max_abs": bounded_weight_max_abs,
            "gradient_max_abs": gradient_max_abs,
            "gradient_relative_l2": gradient_relative_l2,
            "bounded_gradient_max_abs": bounded_gradient_max_abs,
            "bounded_gradient_relative_l2": bounded_gradient_relative_l2,
        },
        "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
