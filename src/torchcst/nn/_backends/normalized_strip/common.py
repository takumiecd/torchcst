"""Minimal shared support routing for normalized Strip kernels."""

import math

import torch
import triton as tr
import triton.language as tl


def _norm_options(plan, low, high):
    shape = tuple(
        tr.next_power_of_2(math.ceil(2 * high * 1.00001 / s) + 1) for s in plan.spacing
    )
    return dict(
        A=None,
        N=plan.sizes[0],
        H=plan.sizes[1],
        J=plan.sizes[2],
        **dict(zip(("O0", "O1", "O2"), plan.origin)),
        **dict(zip(("S0", "S1", "S2"), plan.spacing)),
        LO=math.log(low),
        HI=math.log(high),
        B0=shape[0],
        B1=shape[1],
        B2=shape[2],
        B=tr.next_power_of_2(math.prod(shape)),
    )


def validate_norm_plan(plan):
    from ..cuda.algorithms.normalized_strip.constraints import routing_reasons

    reasons = routing_reasons(plan)
    if reasons:
        raise ValueError(reasons[0])


def ball_offsets(plan, device):
    if plan.spacing != (1.0, 0.5, 0.5):
        raise ValueError("compact ball normalizer supports common spacing(1,.5,.5)")
    offsets = getattr(plan, "_norm_ball_offsets", None)
    if offsets is None:
        import itertools

        packed = []
        for i, j, k in itertools.product(range(-4, 5), range(-8, 9), range(-8, 9)):
            # Minimum squared distance from the nearest-cell fractional box.
            lower = (
                max(abs(i) - 0.5, 0.0) ** 2
                + max(0.5 * abs(j) - 0.25, 0.0) ** 2
                + max(0.5 * abs(k) - 0.25, 0.0) ** 2
            )
            if lower <= 3.25**2 * 1.00001:
                packed.append((i + 4) | ((j + 8) << 4) | ((k + 8) << 9))
        offsets = torch.tensor(packed, dtype=torch.int32, device=device)
        plan._norm_ball_offsets = offsets
    return offsets


@tr.jit
def _owners(
    P,
    Keys,
    A: tl.constexpr,
    O: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    a = tl.program_id(0) * B + tl.arange(0, B)
    c = tl.load(P + a * 5 + 2, a < A, O)
    key = tl.minimum(tl.maximum(tl.floor((c - O) / (32.0 * S)), 0), G - 1)
    tl.store(Keys + a, key.to(tl.int32), a < A)


@tr.jit
def _bucket_histogram(Keys, Counts, A: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    key = tl.load(Keys + atom, valid, 0)
    tl.atomic_add(Counts + key, 1, valid, sem="relaxed")


@tr.jit
def _bucket_scatter(Keys, Cursors, SortedKeys, Order, A: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    valid = atom < A
    key = tl.load(Keys + atom, valid, 0)
    position = tl.atomic_add(Cursors + key, 1, valid, sem="relaxed")
    tl.store(SortedKeys + position, key, valid)
    tl.store(Order + position, atom, valid)


@torch.no_grad()
def atomic_bucket_sort(keys, buckets):
    """Forward-only low-scratch bucket permutation; order within a bucket varies."""
    if keys.dtype != torch.int32 or keys.device.type != "cuda":
        raise ValueError("atomic bucket sort requires CUDA int32 keys")
    counts = torch.zeros(buckets, dtype=torch.int32, device=keys.device)
    if keys.numel():
        _bucket_histogram[(tr.cdiv(keys.numel(), 128),)](
            keys, counts, keys.numel(), 128, num_warps=4
        )
    cursors = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts
    sorted_keys = torch.empty_like(keys)
    order = torch.empty(keys.numel(), dtype=torch.int32, device=keys.device)
    if keys.numel():
        _bucket_scatter[(tr.cdiv(keys.numel(), 128),)](
            keys, cursors, sorted_keys, order, keys.numel(), 128, num_warps=4
        )
    return sorted_keys, order
