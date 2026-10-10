"""Untimed work census and controlled Y probes; never a registered executor.

FULL, sorted metadata and scale-first retain the real formula. Other modes
are diagnostic lower bounds with changed outputs, not selectable Algorithms.
"""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _periodic_raw


@tr.jit
def aggregation_probe(
    P,
    Order,
    Bounds,
    MaxDistance,
    H,
    Y,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    LO: tl.constexpr,
    OO: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    BO: tl.constexpr,
    BSTART: tl.constexpr,
    SORTED: tl.constexpr = False,
    SCALE_FIRST: tl.constexpr = False,
    MODE: tl.constexpr = "full",
    KEEP_ID_LOAD: tl.constexpr = False,
):
    owner = tl.program_id(0)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    sites = owner * BO + tl.arange(0, BO)
    bins: tl.constexpr = tr.cdiv(NO, BO)
    distance = tl.load(MaxDistance)
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
            if SORTED and not KEEP_ID_LOAD:
                a = pos
            else:
                a = tl.load(Order + pos, valid, 0).to(tl.int32)
            if MODE == "index-walk":
                # Real index loads remain observable; lower bound, not real Y.
                accumulator += tl.sum(tl.where(valid, a + 1, 0), 0).to(tl.float32)
            else:
                if MODE == "gather-reduce":
                    # Cheap varying coefficients preserve GROUP/BM/BO axes.
                    coefficient = (a[:, None] & 7).to(tl.float32) + sites[None, :].to(
                        tl.float32
                    ) * 0.001
                    contributes = valid
                else:
                    co = tl.load(P + 3 * A + a, valid, OO)
                    inv = tl.load(P + A + a, valid, 1)
                    norm = tl.load(P + 5 * A + a, valid, 1)
                    amp = tl.load(P + a, valid, 0)
                    if MODE == "norm-one":
                        # Bypasses variable normalization division. Changes Y;
                        # never an executable mathematical variant or Plan.
                        norm = tl.full((GROUP,), 1.0, tl.float32)
                    u, _ = _periodic_raw(
                        sites[None, :], co[:, None], inv[:, None], OO, LO, NO
                    )
                    u = tl.where(valid[:, None] & (sites[None, :] < NO), u, 0.0)
                    contributes = valid & (tl.max(u, 1) > 0)
                    if MODE == "support-unit":
                        u = (u > 0).to(tl.float32)
                    if SCALE_FIRST:
                        coefficient = u * tl.div_rn(amp, norm)[:, None]
                    else:
                        coefficient = tl.div_rn(u, norm[:, None]) * amp[:, None]
                if MODE == "synthetic-h":
                    # Removes H references, keeps varying atom/batch values.
                    h = (pos[:, None] + rows[None, :] + 1).to(tl.float32) * 0.00001
                    h = tl.where(contributes[:, None] & (rows[None, :] < B), h, 0.0)
                else:
                    h = tl.load(
                        H
                        + (tl.program_id(1) * A + pos[:, None]) * BM
                        + tl.arange(0, BM)[None, :],
                        contributes[:, None] & (rows[None, :] < B),
                        0.0,
                    )
                accumulator += tl.sum(h[:, :, None] * coefficient[:, None, :], 0)
    tl.store(
        Y + rows[:, None] * NO + sites[None, :],
        accumulator,
        (rows[:, None] < B) & (sites[None, :] < NO),
    )


@tr.jit
def work_census(
    P,
    Order,
    Bounds,
    MaxDistance,
    Counts,
    A: tl.constexpr,
    NO: tl.constexpr,
    LO: tl.constexpr,
    OO: tl.constexpr,
    GROUP: tl.constexpr,
    BO: tl.constexpr,
):
    owner = tl.program_id(0)
    sites = owner * BO + tl.arange(0, BO)
    bins: tl.constexpr = tr.cdiv(NO, BO)
    radius = (tl.load(MaxDistance) + BO - 1) // BO + (1 if NO % BO else 0)
    count = tl.minimum(2 * radius + 1, bins)
    first = tl.where(count == bins, 0, owner - radius)
    lane = tl.arange(0, GROUP)
    groups = tl.full((), 0, tl.int32)
    candidates = tl.full((), 0, tl.int32)
    active = tl.full((), 0, tl.int32)
    pairs = tl.full((), 0, tl.int32)
    sectors = tl.full((), 0, tl.int32)
    sorted_sectors = tl.full((), 0, tl.int32)
    for neighbor in range(count):
        bucket = ((first + neighbor) % bins + bins) % bins
        low = tl.load(Bounds + bucket).to(tl.int32)
        high = tl.load(Bounds + bucket + 1).to(tl.int32)
        for start in range(low, high, GROUP):
            pos = start + lane
            valid = pos < high
            a = tl.load(Order + pos, valid, 0).to(tl.int32)
            co = tl.load(P + 3 * A + a, valid, OO)
            inv = tl.load(P + A + a, valid, 1)
            u, _ = _periodic_raw(sites[None, :], co[:, None], inv[:, None], OO, LO, NO)
            positive = valid[:, None] & (sites[None, :] < NO) & (u > 0)
            groups += 1
            candidates += tl.sum(valid.to(tl.int32), 0)
            active += tl.sum((tl.max(positive.to(tl.int32), 1) > 0).to(tl.int32), 0)
            pairs += tl.sum(tl.sum(positive.to(tl.int32), 1), 0)
            # Logical 32-byte sector cardinality per 8-lane load. These are
            # address counts, not measured transactions, cache hits or misses.
            earlier = lane[None, :] < lane[:, None]
            has_prior = earlier & valid[None, :]
            same = (a[:, None] // 8) == (a[None, :] // 8)
            sectors += tl.sum(
                (valid & (tl.sum((has_prior & same).to(tl.int32), 1) == 0)).to(
                    tl.int32
                ),
                0,
            )
            sorted_same = (pos[:, None] // 8) == (pos[None, :] // 8)
            sorted_sectors += tl.sum(
                (valid & (tl.sum((has_prior & sorted_same).to(tl.int32), 1) == 0)).to(
                    tl.int32
                ),
                0,
            )
    offset = owner * 7
    tl.store(Counts + offset, count)
    tl.store(Counts + offset + 1, groups)
    tl.store(Counts + offset + 2, candidates)
    tl.store(Counts + offset + 3, active)
    tl.store(Counts + offset + 4, pairs)
    tl.store(Counts + offset + 5, sectors)
    tl.store(Counts + offset + 6, sorted_sectors)
