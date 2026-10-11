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
    OUTPUT: tl.constexpr = True,
    SITE_ROUTING: tl.constexpr = False,
    SECONDARY_OUTPUT: tl.constexpr = False,
    SECONDARY_N: tl.constexpr = 1,
    SECONDARY_L: tl.constexpr = 1.0,
    SECONDARY_O: tl.constexpr = 0.0,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    co = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, OO)
    phase = co - OO - LO * tl.floor(tl.div_rn(co - OO, LO))
    site = tl.minimum(tl.maximum(tl.floor(tl.div_rn(phase, LO / NO)), 0), NO - 1).to(
        tl.int32
    )
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    # Bound distance to every prepared unwrapped support site. Unlike a fixed
    # sigma cap this remains exact for live/broad/fallback supports, including
    # precision guards. Empty supports need no routing radius.
    distance = tl.maximum(tl.abs(low - site), tl.abs(high - 1 - site)) + 2
    distance = tl.where(valid & (high > low), distance, 0)
    key = site if SITE_ROUTING else site // BO
    if SECONDARY_OUTPUT:
        tl.static_assert(not OUTPUT and not SITE_ROUTING)
        # Input coarse bin stays primary; output centre only orders its members.
        # Full supported shape is <=8192 per side, hence int32 key is safe.
        centre = tl.load(P + 3 * A + a, valid, SECONDARY_O)
        phase2 = (
            centre
            - SECONDARY_O
            - SECONDARY_L * tl.floor(tl.div_rn(centre - SECONDARY_O, SECONDARY_L))
        )
        secondary = tl.minimum(
            tl.maximum(tl.floor(tl.div_rn(phase2, SECONDARY_L / SECONDARY_N)), 0),
            SECONDARY_N - 1,
        ).to(tl.int32)
        key = key * SECONDARY_N + secondary
    tl.store(Keys + a, key, valid)
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
    PROFILE_OUTPUT: tl.constexpr = True,
    SITE_ROUTING: tl.constexpr = False,
    H_BM: tl.constexpr = 0,
):
    owner = tl.program_id(0)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    sites = owner * BO + tl.arange(0, BO)
    bins: tl.constexpr = tr.cdiv(NO, BO)
    distance = tl.load(MaxDistance)
    if SITE_ROUTING:
        # Expand the actual (possibly short final) owner interval by the same
        # guarded support distance as coarse routing. Centre keys in this
        # circular interval occupy at most two contiguous sorted-position
        # ranges, with no duplicated atoms even for full-axis support.
        width = tl.minimum(BO, NO - owner * BO)
        site_count = tl.minimum(width + 2 * distance, NO)
        first = tl.where(site_count == NO, 0, ((owner * BO - distance) % NO + NO) % NO)
        end = first + site_count
        count = tl.where(end > NO, 2, 1)
    else:
        # One extra bin covers the short final bin on a non-multiple grid size.
        radius = (distance + BO - 1) // BO + (1 if NO % BO else 0)
        count = tl.minimum(2 * radius + 1, bins)
        first = tl.where(count == bins, 0, owner - radius)
    accumulator = tl.full((BM, BO), 0, tl.float32)
    lane = tl.arange(0, GROUP)
    for neighbor in range(count):
        if SITE_ROUTING:
            begin_site = tl.where(neighbor == 0, first, 0)
            end_site = tl.where(neighbor == 0, tl.minimum(end, NO), end - NO)
            low = tl.load(Bounds + begin_site).to(tl.int32)
            high = tl.load(Bounds + end_site).to(tl.int32)
        else:
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
                co = tl.load(P + (3 if PROFILE_OUTPUT else 2) * A + a, valid, OO)
                inv = tl.load(P + A + a, valid, 1)
                norm = tl.load(P + (5 if PROFILE_OUTPUT else 4) * A + a, valid, 1)
                amp = tl.load(P + a, valid, 0)
            u, _ = _periodic_raw(sites[None, :], co[:, None], inv[:, None], OO, LO, NO)
            u = tl.where(valid[:, None] & (sites[None, :] < NO), u, 0.0)
            contributes = valid & (tl.max(u, 1) > 0)
            if CACHED:
                # H is stored by sorted position, so owners load consecutive
                # atom/batch values without another original-ID gather.
                if H_BM:
                    # Owner rows cross producer slabs. H remains [slab,A,H_BM].
                    local_rows = tl.program_id(1) * BM + tl.arange(0, BM)
                    offsets = (
                        local_rows[None, :] // H_BM * A + pos[:, None]
                    ) * H_BM + local_rows[None, :] % H_BM
                else:
                    offsets = (tl.program_id(1) * A + pos[:, None]) * BM + tl.arange(
                        0, BM
                    )[None, :]
                h = tl.load(
                    H + offsets,
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
def prepare_output_fields(
    P,
    Order,
    Hot,
    Unsafe,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
    OUTPUT: tl.constexpr = True,
):
    """Ephemeral SoA beta/inverse-width/centre, sorted identically to raw H."""
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    amp = tl.load(P + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    inv = tl.load(P + A + a, valid, 1)
    co = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    beta = tl.div_rn(amp, norm)
    unsafe = tl.max((valid & ~(tl.abs(beta) < float("inf"))).to(tl.int32), 0)
    # No atomic is issued on ordinary finite scales. One forward flag covers
    # exceptional atoms without adding fields or per-owner ID dependencies.
    tl.atomic_or(Unsafe, unsafe, unsafe != 0, sem="relaxed")
    tl.store(Hot + pos, beta, valid)
    tl.store(Hot + A + pos, inv, valid)
    tl.store(Hot + 2 * A + pos, co, valid)


@tr.jit
def prepared_output_owned(
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
    H,
    Hot,
    Unsafe,
    BSTART: tl.constexpr,
    FALLBACK: tl.constexpr,
    PROFILE_OUTPUT: tl.constexpr = True,
    SITE_ROUTING: tl.constexpr = False,
    H_BM: tl.constexpr = 0,
):
    # Separate specializations keep the original-ID/division path out of the
    # fast kernel's register schedule. Only one of the two launches writes Y.
    if (tl.load(Unsafe) != 0) == FALLBACK:
        output_owned(
            X,
            P,
            Order,
            Bounds,
            MaxDistance,
            Y,
            A,
            B,
            NI,
            NO,
            LI,
            LO,
            OI,
            OO,
            X0,
            X1,
            BM,
            BK,
            GROUP,
            BO,
            H=H,
            BSTART=BSTART,
            CACHED=True,
            Hot=Hot,
            PREPARED=not FALLBACK,
            PROFILE_OUTPUT=PROFILE_OUTPUT,
            SITE_ROUTING=SITE_ROUTING,
            H_BM=H_BM,
        )
