"""Joint centre cells: register H/G, disjoint partials, deterministic reduction.

Small supports use one cell and a bounded halo. Broad supports use exact
owner recomputation in the same bounded scratch, never truncate support.
"""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _periodic_raw
from .kernels import _contract


@tr.jit
def _centre_bin(
    P,
    a,
    valid,
    A: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
):
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, O)
    phase = center - O - L * tl.floor(tl.div_rn(center - O, L))
    site = tl.minimum(tl.maximum(tl.floor(tl.div_rn(phase, L / N)), 0), N - 1).to(
        tl.int32
    )
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    distance = tl.maximum(tl.abs(low - site), tl.abs(high - 1 - site)) + 2
    return site // TILE, tl.where(valid & (high > low), distance, 0)


@tr.jit
def joint_keys(
    P,
    Keys,
    Distances,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    ib, di = _centre_bin(P, a, valid, A, NI, LI, OI, TILE, False)
    ob, do = _centre_bin(P, a, valid, A, NO, LO, OO, TILE, True)
    tl.store(Keys + a, ob * tr.cdiv(NI, TILE) + ib, valid)
    blocks: tl.constexpr = tr.cdiv(A, BLOCK)
    tl.store(Distances + tl.program_id(0), tl.max(di, 0))
    tl.store(Distances + blocks + tl.program_id(0), tl.max(do, 0))


@tr.jit
def _halo_enabled(Distance, NI: tl.constexpr, NO: tl.constexpr, TILE: tl.constexpr):
    if NI % TILE == 0 and NO % TILE == 0 and NI >= 3 * TILE and NO >= 3 * TILE:
        return (tl.load(Distance) <= TILE) & (tl.load(Distance + 1) <= TILE)
    else:
        return tl.full((), False, tl.int1)


@tr.jit
def joint_local(
    X,
    DY,
    P,
    Order,
    Bounds,
    Distance,
    Scratch,
    Partial,
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
    D0: tl.constexpr,
    D1: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
    BACKWARD: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
):
    cell = tl.program_id(0)
    nbi: tl.constexpr = tr.cdiv(NI, TILE)
    nbo: tl.constexpr = tr.cdiv(NO, TILE)
    ob, ib = cell // nbi, cell % nbi
    rows = BSTART + tl.arange(0, BM)
    k = tl.arange(0, 4 * TILE)
    lane = tl.arange(0, GROUP)
    fast = _halo_enabled(Distance, NI, NO, TILE)
    if BACKWARD:
        owner, bins, n, length, origin = ib, nbi, NI, LI, OI
        distance = tl.load(Distance)
    else:
        owner, bins, n, length, origin = ob, nbo, NO, LO, OO
        distance = tl.load(Distance + 1)
    sites = owner * TILE + k - tl.where(fast, TILE, 0)
    active = (k < tl.where(fast, 3 * TILE, TILE)) & tl.where(fast, True, sites < n)
    accumulator = tl.full((BM, 4 * TILE), 0.0, tl.float32)
    radius = (distance + TILE - 1) // TILE + (1 if n % TILE else 0)
    count = tl.where(fast, 1, tl.minimum(2 * radius + 1, bins))
    first = tl.where(fast | (count == bins), owner if fast else 0, owner - radius)
    for neighbor in range(count):
        bucket = ((first + neighbor) % bins + bins) % bins
        key = ob * nbi + bucket if BACKWARD else bucket * nbi + ib
        low = tl.load(Bounds + key).to(tl.int32)
        high = tl.load(Bounds + key + 1).to(tl.int32)
        for start in range(low, high, GROUP):
            pos = start + lane
            valid = pos < high
            a = tl.load(Order + pos, valid, 0).to(tl.int32)
            center = tl.load(P + (2 if BACKWARD else 3) * A + a, valid, origin)
            inv = tl.load(P + A + a, valid, 1)
            norm = tl.load(P + (4 if BACKWARD else 5) * A + a, valid, 1)
            amp = tl.load(P + a, valid, 0)
            q, _ = _periodic_raw(
                sites[None, :], center[:, None], inv[:, None], origin, length, n
            )
            q = tl.where(valid[:, None] & active[None, :], q, 0.0)
            contributes = valid & (tl.max(q, 1) > 0)
            if BACKWARD:
                # Every atom belongs to exactly one centre cell, also when
                # either support is empty. That cell owns its parameter VJP.
                writer = valid & (fast | (bucket == ib))
                enabled = contributes | (writer if NEED_P else False)
                g, dg = _contract(
                    DY,
                    P,
                    a,
                    rows,
                    enabled,
                    A,
                    B,
                    NO,
                    LO,
                    OO,
                    D0,
                    D1,
                    True,
                    NEED_P,
                    BM,
                    BK,
                    GROUP,
                    True,
                )
                if NEED_X:
                    coefficient = tl.div_rn(q, norm[:, None]) * amp[:, None]
                    accumulator += tl.sum(g[:, :, None] * coefficient[:, None, :], 0)
                if NEED_P:  # noqa: SIM102 - constexpr keeps writer VJP specialization.
                    if tl.sum(writer.to(tl.int32), 0) > 0:
                        h, dh = _contract(
                            X,
                            P,
                            a,
                            rows,
                            writer,
                            A,
                            B,
                            NI,
                            LI,
                            OI,
                            X0,
                            X1,
                            False,
                            True,
                            BM,
                            BK,
                            GROUP,
                            True,
                        )
                        tl.store(Partial + a, tl.sum(h * g, 1), writer)
                        tl.store(Partial + A + a, amp * tl.sum(g * dh, 1), writer)
                        tl.store(Partial + 2 * A + a, amp * tl.sum(h * dg, 1), writer)
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
                coefficient = tl.div_rn(q, norm[:, None]) * amp[:, None]
                accumulator += tl.sum(h[:, :, None] * coefficient[:, None, :], 0)
    if not BACKWARD or NEED_X:
        tl.store(
            Scratch + (cell * BM + tl.arange(0, BM)[:, None]) * (4 * TILE) + k[None, :],
            accumulator,
        )


@tr.jit
def reduce_joint(
    Scratch,
    Distance,
    Result,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    BSTART: tl.constexpr,
    BM: tl.constexpr,
    TILE: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    owner = tl.program_id(0)
    nbi: tl.constexpr = tr.cdiv(NI, TILE)
    nbo: tl.constexpr = tr.cdiv(NO, TILE)
    n: tl.constexpr = NI if BACKWARD else NO
    bins: tl.constexpr = nbi if BACKWARD else nbo
    other_bins: tl.constexpr = nbo if BACKWARD else nbi
    rows = BSTART + tl.arange(0, BM)
    j = owner * TILE + tl.arange(0, TILE)
    fast = _halo_enabled(Distance, NI, NO, TILE)
    acc = tl.full((BM, TILE), 0.0, tl.float32)
    for neighbor in range(tl.where(fast, 3, 1)):
        center_bin = (owner + neighbor - tl.where(fast, 1, 0) + bins) % bins
        local = j - center_bin * TILE + TILE
        local = (local % n + n) % n
        local = tl.where(fast, local, j - owner * TILE)
        for other in range(other_bins):
            cell = other * nbi + center_bin if BACKWARD else center_bin * nbi + other
            values = tl.load(
                Scratch
                + (cell * BM + tl.arange(0, BM)[:, None]) * (4 * TILE)
                + local[None, :]
            )
            acc += values
    tl.store(
        Result + rows[:, None] * n + j[None, :],
        acc,
        (rows[:, None] < B) & (j[None, :] < n),
    )
