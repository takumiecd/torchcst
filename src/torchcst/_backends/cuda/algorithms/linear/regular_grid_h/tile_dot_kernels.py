"""Support-tile memberships and IEEE FP32 matrix contractions.

At most two memberships per axis in the fast path. Wider supports and short
edge tiles use the exact all-atom fallback; nothing is clipped or normalized
per tile. Every H/G writer owns an atom, every output writer owns a tile part.
"""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _periodic_raw
from .kernels import _contract


@tr.jit
def membership(
    P,
    Keys,
    Wide,
    A: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    OUTPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    live = (high > low) & (other_high > other_low)
    bins: tl.constexpr = tr.cdiv(N, TILE)
    first = ((tl.floor(low.to(tl.float32) / TILE).to(tl.int32)) % bins + bins) % bins
    last = (
        (tl.floor((high - 1).to(tl.float32) / TILE).to(tl.int32)) % bins + bins
    ) % bins
    first = tl.where(live, first, 0)
    second = live & (last != first)
    tl.store(Keys + 2 * a, first, valid)
    tl.store(Keys + 2 * a + 1, tl.where(second, last, bins), valid)
    wide = live & ((high - low) > TILE)
    if N % TILE != 0:
        wide = valid
    tl.store(Wide + tl.program_id(0), tl.max(wide.to(tl.int32), 0))


@tr.jit
def coefficients(
    P,
    a,
    valid,
    j,
    A: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
):
    inv = tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    q, dq = _periodic_raw(j[None, :], center[:, None], inv[:, None], O, L, N)
    mask = valid[:, None] & (j[None, :] < N) & (other_high > other_low)[:, None]
    q = tl.div_rn(tl.where(mask, q, 0.0), norm[:, None])
    if DERIVATIVE:
        loggrad = tl.load(P + (7 if OUTPUT else 6) * A + a, valid, 0)
        flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
        dq = tl.div_rn(tl.where(mask, dq, 0.0), norm[:, None])
        dq = tl.where(
            (flags[:, None] & (2 if OUTPUT else 1)) != 0, 0.0, dq - loggrad[:, None] * q
        )
    return q, dq


@tr.jit
def produce(
    X,
    P,
    Order,
    Wide,
    H,
    DH,
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
    GROUP: tl.constexpr = 32,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    rows = START + tl.arange(0, 16)
    if tl.load(Wide) == 0:
        entry = tl.load(Order + pos, pos < 2 * A, 1).to(tl.int32)
        a = entry // 2
        valid = (pos < 2 * A) & ((entry % 2) == 0)
        low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
        high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
        olow = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
        ohigh = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
        live = (high > low) & (ohigh > olow)
        bins: tl.constexpr = tr.cdiv(N, TILE)
        first = (
            (tl.floor(low.to(tl.float32) / TILE).to(tl.int32)) % bins + bins
        ) % bins
        last = (
            (tl.floor((high - 1).to(tl.float32) / TILE).to(tl.int32)) % bins + bins
        ) % bins
        first = tl.where(live, first, 0)
        second = live & (last != first)
        acc = tl.full((16, GROUP), 0.0, tl.float32)
        grad = tl.full((16, GROUP), 0.0, tl.float32)
        begin = tl.min(tl.where(valid, first, bins), 0)
        end = tl.max(tl.where(valid, first + 1, 0), 0)
        for tile in range(begin, end):
            for offset in range(2):
                j = ((tile + offset) % bins) * TILE + tl.arange(0, TILE)
                enabled = valid & (first == tile) & ((offset == 0) | second)
                q, dq = coefficients(P, a, enabled, j, A, N, L, O, OUTPUT, DERIVATIVE)
                xx = tl.load(
                    X + rows[:, None] * X0 + j[None, :] * X1,
                    (rows[:, None] < B) & (tl.arange(0, 16)[:, None] < BM),
                    0.0,
                )
                acc += tl.dot(xx, tl.trans(q), input_precision="ieee")
                if DERIVATIVE:
                    grad += tl.dot(xx, tl.trans(dq), input_precision="ieee")
        tl.store(
            H + a[None, :] * BM + tl.arange(0, 16)[:, None],
            acc,
            valid[None, :] & (rows[:, None] < B) & (tl.arange(0, 16)[:, None] < BM),
        )
        if DERIVATIVE:
            tl.store(
                DH + a[None, :] * BM + tl.arange(0, 16)[:, None],
                grad,
                valid[None, :] & (rows[:, None] < B) & (tl.arange(0, 16)[:, None] < BM),
            )


@tr.jit
def fallback_produce(
    X,
    P,
    Wide,
    H,
    DH,
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
    GROUP: tl.constexpr = 32,
):
    # Separate compilation keeps the large exact broad-support temporary out
    # of the narrow GEMM producer's register allocation. GPU flag remains live.
    if tl.load(Wide) != 0:
        pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
        rows = START + tl.arange(0, 16)
        a = pos
        valid = a < A
        h, dh = _contract(
            X,
            P,
            a,
            rows,
            valid,
            A,
            B,
            N,
            L,
            O,
            X0,
            X1,
            OUTPUT,
            DERIVATIVE,
            16,
            32,
            GROUP,
        )
        mask = valid[:, None] & (rows[None, :] < B) & (tl.arange(0, 16)[None, :] < BM)
        tl.store(H + a[:, None] * BM + tl.arange(0, 16)[None, :], h, mask)
        if DERIVATIVE:
            tl.store(DH + a[:, None] * BM + tl.arange(0, 16)[None, :], dh, mask)


@tr.jit
def owner(
    H,
    P,
    Order,
    Bounds,
    Wide,
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
    GROUP: tl.constexpr = 32,
):
    tile, part = tl.program_id(0), tl.program_id(1)
    j = tile * TILE + tl.arange(0, TILE)
    k = tl.arange(0, GROUP)
    r = tl.arange(0, 16)
    fast = tl.load(Wide) == 0
    low = tl.load(Bounds + tile).to(tl.int32) if fast else 0
    high = tl.load(Bounds + tile + 1).to(tl.int32) if fast else A
    blocks = (high - low + GROUP - 1) // GROUP
    per_part = (blocks + PARTS - 1) // PARTS
    begin = low + part * per_part * GROUP
    end = tl.minimum(high, begin + per_part * GROUP)
    acc = tl.full((16, TILE), 0.0, tl.float32)
    for start in range(begin, end, GROUP):
        pos = start + k
        valid = pos < end
        a = tl.load(Order + pos, valid, 0).to(tl.int32) // 2 if fast else pos
        q, _ = coefficients(P, a, valid, j, A, N, L, O, OUTPUT, False)
        amp = tl.load(P + a, valid, 0)
        q = q * amp[:, None]
        h = tl.load(
            H + a[None, :] * BM + r[:, None],
            valid[None, :] & (START + r[:, None] < B) & (r[:, None] < BM),
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


@tr.jit
def reduce_owner(
    Scratch,
    Result,
    B: tl.constexpr,
    N: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    TILE: tl.constexpr,
    PARTS: tl.constexpr,
):
    tile = tl.program_id(0)
    r = tl.arange(0, BM)
    k = tl.arange(0, TILE)
    acc = tl.full((BM, TILE), 0.0, tl.float32)
    for part in tl.static_range(PARTS):
        acc += tl.load(
            Scratch + ((tile * PARTS + part) * BM + r[:, None]) * TILE + k[None, :]
        )
    tl.store(
        Result + (START + r[:, None]) * N + tile * TILE + k[None, :],
        acc,
        (START + r[:, None] < B) & (tile * TILE + k[None, :] < N),
    )


@tr.jit
def physical(
    H,
    DH,
    G,
    DG,
    P,
    Physical,
    A: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, BM)
    mask = (a[:, None] < A) & (START + r[None, :] < B)
    h = tl.load(H + a[:, None] * BM + r[None, :], mask, 0.0)
    dh = tl.load(DH + a[:, None] * BM + r[None, :], mask, 0.0)
    g = tl.load(G + a[:, None] * BM + r[None, :], mask, 0.0)
    dg = tl.load(DG + a[:, None] * BM + r[None, :], mask, 0.0)
    amp = tl.load(P + a, a < A, 0.0)
    da = tl.sum(h * g, 1)
    di = amp * tl.sum(dh * g, 1)
    do = amp * tl.sum(h * dg, 1)
    for field in tl.static_range(3):
        value = da if field == 0 else (di if field == 1 else do)
        if START != 0:
            value += tl.load(Physical + field * A + a, a < A, 0.0)
        tl.store(Physical + field * A + a, value, a < A)
