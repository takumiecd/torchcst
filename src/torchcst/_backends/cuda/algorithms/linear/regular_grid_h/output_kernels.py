"""Exact output ownership; no atomic writes to Y and no full H tensor."""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _periodic_raw
from .kernels import _contract


@tr.jit
def routing_keys(
    P,
    Keys,
    Distances,
    A: tl.constexpr,
    NO: tl.constexpr,
    LO: tl.constexpr,
    OO: tl.constexpr,
    BO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    co = tl.load(P + 3 * A + a, valid, OO)
    phase = co - OO - LO * tl.floor(tl.div_rn(co - OO, LO))
    site = tl.minimum(tl.maximum(tl.floor(tl.div_rn(phase, LO / NO)), 0), NO - 1).to(
        tl.int32
    )
    low = tl.load(P + 11 * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + 12 * A + a, valid, 0).to(tl.int32)
    # Bound distance to every prepared unwrapped support site. Unlike a fixed
    # sigma cap this remains exact for live/broad/fallback supports, including
    # precision guards. Empty supports need no routing radius.
    distance = tl.maximum(tl.abs(low - site), tl.abs(high - 1 - site)) + 2
    distance = tl.where(valid & (high > low), distance, 0)
    tl.store(Keys + a, site // BO, valid)
    tl.store(Distances + tl.program_id(0), tl.max(distance, 0))


@tr.jit
def output_owned(
    X,
    P,
    Order,
    Bounds,
    MaxDistance,
    Y,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    BO: tl.constexpr,
    H=None,
    BSTART: tl.constexpr = 0,
    CACHED: tl.constexpr = False,
    Hot=None,
    PREPARED: tl.constexpr = False,
):
    owner = tl.program_id(0)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    sites = owner * BO + tl.arange(0, BO)
    bins: tl.constexpr = tr.cdiv(NO, BO)
    distance = tl.load(MaxDistance)
    # One extra bin covers the short final bin on a non-multiple grid size.
    radius = (distance + BO - 1) // BO + (1 if NO % BO else 0)
    count = tl.minimum(2 * radius + 1, bins)
    first = tl.where(count == bins, 0, owner - radius)
    accumulator = tl.full((BM, BO), 0, tl.float32)
    lane = tl.arange(0, GROUP)
    for neighbor in range(count):
        bucket = ((first + neighbor) % bins + bins) % bins
        low = tl.load(Bounds + bucket).to(tl.int32)
        high = tl.load(Bounds + bucket + 1).to(tl.int32)
        for start in range(low, high, GROUP):
            pos = start + lane
            valid = pos < high
            if PREPARED:
                # Same position as H: no original-ID dependency in the owner.
                co = tl.load(Hot + 2 * A + pos, valid, OO)
                inv = tl.load(Hot + A + pos, valid, 1)
                beta = tl.load(Hot + pos, valid, 0)
            else:
                a = tl.load(Order + pos, valid, 0).to(tl.int32)
                co = tl.load(P + 3 * A + a, valid, OO)
                inv = tl.load(P + A + a, valid, 1)
                norm = tl.load(P + 5 * A + a, valid, 1)
                amp = tl.load(P + a, valid, 0)
            u, _ = _periodic_raw(sites[None, :], co[:, None], inv[:, None], OO, LO, NO)
            u = tl.where(valid[:, None] & (sites[None, :] < NO), u, 0.0)
            contributes = valid & (tl.max(u, 1) > 0)
            if CACHED:
                # H is stored by sorted position, so owners load consecutive
                # atom/batch values without another original-ID gather.
                h = tl.load(
                    H
                    + (tl.program_id(1) * A + pos[:, None]) * BM
                    + tl.arange(0, BM)[None, :],
                    contributes[:, None] & (rows[None, :] < B),
                    0.0,
                )
            else:
                h, _ = _contract(
                    X,
                    P,
                    a,
                    rows,
                    contributes,
                    A,
                    B,
                    NI,
                    LI,
                    OI,
                    X0,
                    X1,
                    False,
                    False,
                    BM,
                    BK,
                    GROUP,
                )
            if PREPARED:
                coefficient = u * beta[:, None]
            else:
                coefficient = tl.div_rn(u, norm[:, None]) * amp[:, None]
            accumulator += tl.sum(h[:, :, None] * coefficient[:, None, :], 0)
    tl.store(
        Y + rows[:, None] * NO + sites[None, :],
        accumulator,
        (rows[:, None] < B) & (sites[None, :] < NO),
    )


@tr.jit
def produce_h(
    X,
    P,
    Order,
    H,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    LI: tl.constexpr,
    OI: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    a = tl.load(Order + pos, pos < A, 0).to(tl.int32)
    local_rows = tl.arange(0, BM)
    rows = BSTART + tl.program_id(1) * BM + local_rows
    h, _ = _contract(
        X, P, a, rows, pos < A, A, B, NI, LI, OI, X0, X1, False, False, BM, BK, GROUP
    )
    # Every buffer element is overwritten each chunk, including padded rows.
    tl.store(
        H + (tl.program_id(1) * A + pos[:, None]) * BM + local_rows[None, :],
        h,
        pos[:, None] < A,
    )


@tr.jit
def prepare_output_fields(P, Order, Hot, A: tl.constexpr, BLOCK: tl.constexpr):
    """Ephemeral SoA beta/inverse-width/centre, sorted identically to raw H."""
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    amp = tl.load(P + a, valid, 0)
    norm = tl.load(P + 5 * A + a, valid, 1)
    inv = tl.load(P + A + a, valid, 1)
    co = tl.load(P + 3 * A + a, valid, 0)
    tl.store(Hot + pos, tl.div_rn(amp, norm), valid)
    tl.store(Hot + A + pos, inv, valid)
    tl.store(Hot + 2 * A + pos, co, valid)
