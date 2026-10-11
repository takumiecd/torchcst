"""One ID per atom/axis, dyadic support buckets and unique H/G writers."""

import triton as tr
import triton.language as tl

from .tile_dot_kernels import coefficients
from .tile_schedule_kernels import _consume


@tr.jit
def _membership(
    P,
    a,
    valid,
    A: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
):
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    live = valid & (high > low) & (other_high > other_low)
    lo = ((low % N) + N) % N
    span = tl.minimum(tl.maximum(high - low, 0), N)
    bins: tl.constexpr = tr.cdiv(N, TILE)
    first = lo // TILE
    end = lo + span
    count = tl.where(
        end <= N,
        (end - 1) // TILE - first + 1,
        bins - first + (end - N + TILE - 1) // TILE,
    )
    count = tl.minimum(count, bins)
    # Whole-period intervals use every tile once, including a short edge tile.
    first = tl.where(live & (span < N), first, 0)
    count = tl.where(live, tl.where(span >= N, bins, count), 0)
    return first, count, lo, span, live


@tr.jit
def _single_site(P, a, valid, A: tl.constexpr):
    # Both flags require a nonempty one-site axis and an inactive joint floor.
    # Span checks also guard unusual full-period fallback representations.
    flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    ilo = tl.load(P + 9 * A + a, valid, 0).to(tl.int32)
    ihi = tl.load(P + 10 * A + a, valid, 0).to(tl.int32)
    olo = tl.load(P + 11 * A + a, valid, 0).to(tl.int32)
    ohi = tl.load(P + 12 * A + a, valid, 0).to(tl.int32)
    return valid & ((flags & 3) == 3) & (ihi - ilo == 1) & (ohi - olo == 1)


@tr.jit
def keys(
    P,
    Keys,
    A: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK: tl.constexpr,
    SUPPORT_MODES: tl.constexpr = False,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first, count, _, _, live = _membership(P, a, a < A, A, N, TILE, OUTPUT)
    band = tl.full((BLOCK,), 0, tl.int32)
    for level in tl.static_range(1, LEVELS - (1 if SUPPORT_MODES else 0)):
        band += (count > (1 << (level - 1))).to(tl.int32)
    if SUPPORT_MODES:
        band = tl.where(_single_site(P, a, a < A, A), 0, band + 1)
        # Empty atoms sort after all owner ranges, without an expanded index.
        band = tl.where(live, band, LEVELS)
    tl.store(Keys + a, band * tr.cdiv(N, TILE) + first, a < A)


@tr.jit
def produce(
    X,
    P,
    Order,
    H,
    Partner,
    Physical,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    CONSUME: tl.constexpr,
    LEVELS: tl.constexpr,
    GROUP: tl.constexpr = 32,
    SUPPORT_MODES: tl.constexpr = False,
    Direct=None,
    DIRECT0: tl.constexpr = 0,
    DIRECT1: tl.constexpr = 0,
    NI: tl.constexpr = 1,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    rows = START + tl.arange(0, 16)
    first, count, lo, span, live = _membership(P, a, valid, A, N, TILE, OUTPUT)
    single = (
        _single_site(P, a, valid, A)
        if SUPPORT_MODES
        else tl.full((GROUP,), False, tl.int1)
    )
    normal = valid & live & ~single if SUPPORT_MODES else valid
    band = tl.full((GROUP,), 0, tl.int32)
    for level in tl.static_range(1, LEVELS - (1 if SUPPORT_MODES else 0)):
        band += (count > (1 << (level - 1))).to(tl.int32)
    acc = tl.full((16, GROUP), 0.0, tl.float32)
    grad = tl.full((16, GROUP), 0.0, tl.float32)
    bins: tl.constexpr = tr.cdiv(N, TILE)
    # Only boundary groups can cross a bucket. Process each bucket separately
    # so a wide atom does not lengthen the loop of the narrow members.
    band_begin = tl.min(tl.where(normal, band, LEVELS), 0)
    band_end = tl.max(tl.where(normal, band + 1, 0), 0)
    for current_band in range(band_begin, band_end):
        in_band = normal & (band == current_band)
        tile_begin = tl.min(tl.where(in_band, first, bins), 0)
        tile_end = tl.max(tl.where(in_band, first + 1, 0), 0)
        for tile in range(tile_begin, tile_end):
            enabled = in_band & live & (first == tile)
            begin = tl.min(tl.where(enabled, lo, 2 * N), 0)
            end = tl.max(tl.where(enabled, lo + span, 0), 0)
            for panel in range(begin, end, 16):
                unwrapped = panel + tl.arange(0, 16)
                j = unwrapped % N
                q, dq = coefficients(P, a, enabled, j, A, N, L, O, OUTPUT, DERIVATIVE)
                q = tl.where((unwrapped < end)[None, :], q, 0.0)
                # Individual intervals can occupy different phases within a
                # shared union. Mask each interval to visit every site once.
                inside = (unwrapped[None, :] >= lo[:, None]) & (
                    unwrapped[None, :] < (lo + span)[:, None]
                )
                q = tl.where(inside, q, 0.0)
                xx = tl.load(
                    X + rows[:, None] * X0 + j[None, :] * X1, rows[:, None] < B, 0.0
                )
                if CONSUME != 2:
                    acc += tl.dot(xx, tl.trans(q), input_precision="ieee")
                if DERIVATIVE:
                    dq = tl.where(inside & (unwrapped < end)[None, :], dq, 0.0)
                    grad += tl.dot(xx, tl.trans(dq), input_precision="ieee")
    if CONSUME != 2:
        tl.store(
            H + a[None, :] * BM + tl.arange(0, 16)[:, None],
            acc,
            normal[None, :] & (rows[:, None] < B),
        )
    if CONSUME:
        _consume(
            acc, grad, Partner, P, Physical, a, normal, rows, A, B, START, BM, CONSUME
        )

    if SUPPORT_MODES and CONSUME == 1:
        # Direct one-site dA, unique atom writer. H/G remain logical values;
        # neither a global slab nor a centre derivative is read or written.
        ilo = tl.load(P + 9 * A + a, single, 0).to(tl.int32)
        olo = tl.load(P + 11 * A + a, single, 0).to(tl.int32)
        mask = single[None, :] & (rows[:, None] < B)
        xx = tl.load(
            Direct
            + rows[:, None] * DIRECT0
            + (((ilo % NI) + NI) % NI)[None, :] * DIRECT1,
            mask,
            0.0,
        )
        yy = tl.load(
            X + rows[:, None] * X0 + (((olo % N) + N) % N)[None, :] * X1, mask, 0.0
        )
        da = tl.sum(xx * yy, 0)
        if START != 0:
            da += tl.load(Physical + a, single, 0.0)
        tl.store(Physical + a, da, single)


@tr.jit
def owner(
    H,
    P,
    Order,
    Bounds,
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
    GROUP: tl.constexpr = 32,
):
    tile, part = tl.program_id(0), tl.program_id(1)
    j = tile * TILE + tl.arange(0, TILE)
    k = tl.arange(0, GROUP)
    r = tl.arange(0, 16)
    bins: tl.constexpr = tr.cdiv(N, TILE)
    acc = tl.full((16, TILE), 0.0, tl.float32)
    for band in range(LEVELS):
        upper = tl.minimum(1 << band, bins)
        lower = tile - upper + 1
        # At most two sorted ranges, including the periodic wrap. The
        # full-period bucket uses [0,bins) once rather than overlapping ranges.
        for segment in range(2):
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
