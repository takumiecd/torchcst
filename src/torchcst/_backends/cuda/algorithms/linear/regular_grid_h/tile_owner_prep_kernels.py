"""Controlled owner preparation: active bands, exact ranges and membership."""

import triton as tr
import triton.language as tl

from .tile_bucket_kernels import _membership
from .tile_dot_kernels import coefficients


@tr.jit
def prepare_bands(
    Bounds, Bands, BINS: tl.constexpr, LEVELS: tl.constexpr, BLOCK: tl.constexpr
):
    level = tl.arange(0, BLOCK)
    low = tl.load(Bounds + level * BINS, level < LEVELS, 0)
    high = tl.load(Bounds + (level + 1) * BINS, level < LEVELS, 0)
    active = (level < LEVELS) & (high > low)
    rank = tl.cumsum(active.to(tl.int32), 0)
    tl.store(Bands, tl.sum(active.to(tl.int32), 0))
    tl.store(Bands + rank, level, active)


@tr.jit
def prepare_ranges(
    Bounds,
    Ranges,
    N: tl.constexpr,
    TILE: tl.constexpr,
    PARTS: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bins: tl.constexpr = tr.cdiv(N, TILE)
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < bins * PARTS * LEVELS * 2
    segment = idx % 2
    band = (idx // 2) % LEVELS
    part = (idx // (2 * LEVELS)) % PARTS
    tile = idx // (2 * LEVELS * PARTS)
    upper = tl.minimum(1 << band, bins)
    lower = tile - upper + 1
    left = tl.where(segment == 0, tl.maximum(lower, 0), bins + lower)
    right = tl.where(segment == 0, tile + 1, bins)
    left = tl.where(upper == bins, 0, left)
    right = tl.where(upper == bins, tl.where(segment == 0, bins, 0), right)
    active = (segment == 0) | ((lower < 0) & (upper < bins))
    left = tl.minimum(tl.maximum(left, 0), bins)
    right = tl.minimum(tl.maximum(right, 0), bins)
    low = tl.load(Bounds + band * bins + left, valid, 0)
    high = tl.load(Bounds + band * bins + right, valid, 0)
    high = tl.where(active, high, low)
    blocks = (high - low + 31) // 32
    per_part = (blocks + PARTS - 1) // PARTS
    begin = low + part * per_part * 32
    end = tl.minimum(high, begin + per_part * 32)
    tl.store(Ranges + 2 * idx, begin, valid)
    tl.store(Ranges + 2 * idx + 1, end, valid)


@tr.jit
def prepare_membership(
    P,
    Membership,
    A: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first, count, _, _, _ = _membership(P, a, a < A, A, N, TILE, OUTPUT)
    tl.store(Membership + a, first, a < A)
    tl.store(Membership + A + a, count, a < A)


@tr.jit
def owner(
    H,
    P,
    Order,
    Bounds,
    Bands,
    Ranges,
    Membership,
    Scratch,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    TILE: tl.constexpr,
    PARTS: tl.constexpr,
    OUTPUT: tl.constexpr,
    LEVELS: tl.constexpr,
    PREP_BANDS: tl.constexpr,
    PREP_RANGES: tl.constexpr,
    PREP_MEMBERSHIP: tl.constexpr,
    GROUP: tl.constexpr = 32,
):
    tile, part = tl.program_id(0), tl.program_id(1)
    j = tile * TILE + tl.arange(0, TILE)
    k = tl.arange(0, GROUP)
    r = tl.arange(0, 16)
    bins: tl.constexpr = tr.cdiv(N, TILE)
    acc = tl.full((16, TILE), 0.0, tl.float32)
    band_count = tl.load(Bands) if PREP_BANDS else LEVELS
    for band_index in range(band_count):
        band = tl.load(Bands + 1 + band_index) if PREP_BANDS else band_index
        if not PREP_RANGES:
            upper = tl.minimum(1 << band, bins)
            lower = tile - upper + 1
        for segment in range(2):
            if PREP_RANGES:
                descriptor = (((tile * PARTS + part) * LEVELS + band) * 2 + segment) * 2
                begin = tl.load(Ranges + descriptor)
                end = tl.load(Ranges + descriptor + 1)
            else:
                left = tl.where(segment == 0, tl.maximum(lower, 0), bins + lower)
                right = tl.where(segment == 0, tile + 1, bins)
                left = tl.where(upper == bins, 0, left)
                right = tl.where(upper == bins, tl.where(segment == 0, bins, 0), right)
                active = (segment == 0) | ((lower < 0) & (upper < bins))
                left = tl.minimum(tl.maximum(left, 0), bins)
                right = tl.minimum(tl.maximum(right, 0), bins)
                low = tl.load(Bounds + band * bins + left)
                high = tl.load(Bounds + band * bins + right)
                high = tl.where(active, high, low)
                blocks = (high - low + GROUP - 1) // GROUP
                per_part = (blocks + PARTS - 1) // PARTS
                begin = low + part * per_part * GROUP
                end = tl.minimum(high, begin + per_part * GROUP)
            for start in range(begin, end, GROUP):
                pos = start + k
                valid = pos < end
                a = tl.load(Order + pos, valid, 0).to(tl.int32)
                if PREP_MEMBERSHIP:
                    first = tl.load(Membership + a, valid, 0)
                    count = tl.load(Membership + A + a, valid, 0)
                else:
                    first, count, _, _, _ = _membership(P, a, valid, A, N, TILE, OUTPUT)
                distance = ((tile - first) % bins + bins) % bins
                valid = valid & (distance < count)
                q, _ = coefficients(P, a, valid, j, A, N, L, O, OUTPUT, False)
                amp = tl.load(P + a, valid, 0)
                q *= amp[:, None]
                h = tl.load(
                    H + a[None, :] * BM + r[:, None],
                    valid[None, :] & (START + r[:, None] < B),
                    0.0,
                )
                acc += tl.dot(h, q, input_precision="ieee")
    tl.store(
        Scratch
        + ((tile * PARTS + part) * BM + r[:, None]) * TILE
        + tl.arange(0, TILE)[None, :],
        acc,
        r[:, None] < BM,
    )
