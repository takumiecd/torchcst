"""Live, graph-safe atom ordering by both clipped axis-support patch counts."""

import torch
import triton as tr
import triton.language as tl

from .kernels import _interval


@tr.jit
def support_keys(
    Meta,
    Origin,
    Spacing,
    Keys,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    SPLIT: tl.constexpr = False,
    CUTOFF: tl.constexpr = 0.5,
    Previous=None,
    Dirty=None,
    TRACK: tl.constexpr = False,
    WIDTH_ONLY: tl.constexpr = False,
    TINY_NATIVE: tl.constexpr = False,
    RANGE_TILE: tl.constexpr = 0,
):
    a = tl.program_id(0) * 128 + tl.arange(0, 128)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    iho = tl.load(Meta + a * 12 + 10, a < A, 1)
    ihi = tl.load(Meta + a * 12 + 11, a < A, 1)
    lo, _ = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, iho)
    _, hi = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, iho)
    li, _ = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, ihi)
    _, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, ihi)
    po = tl.cdiv(hi - lo, BS)
    pi = tl.cdiv(ri - li, BS)
    if WIDTH_ONLY:
        # Placement buckets need not change when only a center crosses a site.
        # Actual clipped support is still freshly evaluated by matrix kernels.
        po = tl.minimum(tl.ceil(2 * radius * iho / BS), tr.cdiv(NO, BS)).to(tl.int32)
        pi = tl.minimum(tl.ceil(2 * radius * ihi / BS), tr.cdiv(NI, BS)).to(tl.int32)
    # The radix exceeds every possible input patch count, including full support.
    key = po * (tr.cdiv(NI, BS) + 1) + pi
    if WIDTH_ONLY:
        # Both patch counts are monotone in the same radius and fixed spacings.
        # Their sum distinguishes every reachable pair without a quadratic bank.
        key = po + pi
    if SPLIT:
        offset: tl.constexpr = (
            tr.cdiv(NO, BS) + tr.cdiv(NI, BS) + 1
            if WIDTH_ONLY
            else (tr.cdiv(NO, BS) + 1) * (tr.cdiv(NI, BS) + 1)
        )
        if TRACK or WIDTH_ONLY:
            native = radius <= 2.449489742783178 * CUTOFF
            if TINY_NATIVE:
                native |= tl.minimum(hi - lo, ri - li) <= 2
            broad_key = tl.full((128,), offset, tl.int32)
            if RANGE_TILE:
                # Quantized keys change execution order only, never widths/support.
                width = tl.floor(tl.log2((tl.maximum(po, pi) + 1).to(tl.float32))).to(
                    tl.int32
                )
                tile_o = tl.minimum(lo // RANGE_TILE, tr.cdiv(NO, RANGE_TILE) - 1)
                tile_i = tl.minimum(li // RANGE_TILE, tr.cdiv(NI, RANGE_TILE) - 1)
                broad_key += (width * tr.cdiv(NO, RANGE_TILE) + tile_o) * tr.cdiv(
                    NI, RANGE_TILE
                ) + tile_i
            key = tl.where(native, key, broad_key)
        else:
            key += tl.where(radius <= 2.449489742783178 * CUTOFF, 0, offset)
    if TRACK:
        old = tl.load(Previous + a, a < A, -1)
        changed = tl.sum(((key != old) & (a < A)).to(tl.int32), 0)
        if changed > 0:
            tl.atomic_or(Dirty, 1, sem="relaxed")
        tl.store(Previous + a, key, a < A)
    tl.store(Keys + a, key, a < A)


def order_atoms(meta, origin, spacing, sizes, recipe):
    """Rebuild every forward; only IDs, never Parameters or optimizer states, move."""
    keys = torch.empty(len(meta), dtype=torch.int32, device=meta.device)
    support_keys[(tr.cdiv(len(meta), 128),)](
        meta,
        origin,
        spacing,
        keys,
        len(meta),
        *sizes,
        recipe.matrix_sites,
        recipe.reciprocal,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return torch.argsort(keys)


def partition_atoms(meta, origin, spacing, sizes, recipe):
    """Small IDs first, broad IDs last; a device count keeps Graph shapes fixed."""
    keys = torch.empty(len(meta), dtype=torch.int32, device=meta.device)
    support_keys[(tr.cdiv(len(meta), 128),)](
        meta,
        origin,
        spacing,
        keys,
        len(meta),
        *sizes,
        recipe.matrix_sites,
        recipe.reciprocal,
        True,
        recipe.sigma_cutoff,
        num_warps=4,
        enable_fp_fusion=False,
    )
    radix = (tr.cdiv(sizes[0], recipe.matrix_sites) + 1) * (
        tr.cdiv(sizes[1], recipe.matrix_sites) + 1
    )
    count = (keys < radix).sum(dtype=torch.int32)
    return torch.argsort(keys), count
