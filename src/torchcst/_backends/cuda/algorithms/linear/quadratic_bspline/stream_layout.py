"""Persistent algorithm-local row placement, validated from live support keys."""

import torch
import triton as tr
import triton.language as tl

from .grouping import support_keys


@tr.jit
def _clear(Hist, Cursor, Dirty, K: tl.constexpr):
    if tl.load(Dirty):
        j = tl.program_id(0) * 256 + tl.arange(0, 256)
        tl.store(Hist + j, 0, j < K)
        tl.store(Cursor + j, 0, j < K)


@tr.jit
def _histogram(Keys, Hist, Dirty, A: tl.constexpr):
    if tl.load(Dirty):
        a = tl.program_id(0) * 128 + tl.arange(0, 128)
        key = tl.load(Keys + a, a < A, 0)
        tl.atomic_add(Hist + key, 1, a < A, sem="relaxed")


@tr.jit
def _place(
    Keys,
    Hist,
    Ends,
    Cursor,
    Order,
    Count,
    Dirty,
    Stats,
    A: tl.constexpr,
    K: tl.constexpr,
):
    dirty = tl.load(Dirty)
    if dirty:
        a = tl.program_id(0) * 128 + tl.arange(0, 128)
        key = tl.load(Keys + a, a < A, 0)
        base = tl.load(Ends + key) - tl.load(Hist + key)
        rank = tl.atomic_add(Cursor + key, 1, a < A, sem="relaxed")
        tl.store(Order + base + rank, a.to(tl.int64), a < A)
    if tl.program_id(0) == 0:
        if dirty:
            tl.store(Count, tl.load(Ends + K - 2))
        tl.store(Stats, tl.load(Stats) + 1)
        tl.store(Stats + 1, tl.load(Stats + 1) + dirty.to(tl.int64))


def make_layout(a, sizes, recipe, device):
    radix = (tr.cdiv(sizes[0], recipe.matrix_sites) + 1) * (
        tr.cdiv(sizes[1], recipe.matrix_sites) + 1
    )

    def empty(n, dtype=torch.int32):
        return torch.empty(n, dtype=dtype, device=device)

    return {
        "keys": torch.full((a,), -1, dtype=torch.int32, device=device),
        "dirty": empty(1),
        "hist": empty(radix + 1),
        "ends": empty(radix + 1),
        "cursor": empty(radix + 1),
        "order": empty(a, torch.int64),
        "count": empty(1),
        "stats": torch.zeros(2, dtype=torch.int64, device=device),
    }


def placement(meta, origin, spacing, sizes, recipe, cache):
    if not recipe.reuse_layout:
        support_keys[(tr.cdiv(len(meta), 128),)](
            meta,
            origin,
            spacing,
            cache["keys"],
            len(meta),
            *sizes,
            recipe.matrix_sites,
            recipe.reciprocal,
            True,
            recipe.sigma_cutoff,
            WIDTH_ONLY=True,
            num_warps=4,
            enable_fp_fusion=False,
        )
        count = (cache["keys"] < len(cache["hist"]) - 1).sum(dtype=torch.int32)
        return torch.argsort(cache["keys"]), count
    a = len(meta)
    cache["dirty"].zero_()
    support_keys[(tr.cdiv(a, 128),)](
        meta,
        origin,
        spacing,
        cache["keys"],
        a,
        *sizes,
        recipe.matrix_sites,
        recipe.reciprocal,
        True,
        recipe.sigma_cutoff,
        Previous=cache["keys"],
        Dirty=cache["dirty"],
        TRACK=True,
        WIDTH_ONLY=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k = cache["hist"].numel()
    _clear[(tr.cdiv(k, 256),)](cache["hist"], cache["cursor"], cache["dirty"], k)
    _histogram[(tr.cdiv(a, 128),)](cache["keys"], cache["hist"], cache["dirty"], a)
    # Static-size scan only. Histogram/order writes run exclusively on changed keys.
    torch.cumsum(cache["hist"], 0, dtype=torch.int32, out=cache["ends"])
    _place[(tr.cdiv(a, 128),)](
        cache["keys"],
        cache["hist"],
        cache["ends"],
        cache["cursor"],
        cache["order"],
        cache["count"],
        cache["dirty"],
        cache["stats"],
        a,
        k,
    )
    # Mutable placement may be reused by later forwards before this VJP executes.
    return cache["order"].clone(), cache["count"].clone()
