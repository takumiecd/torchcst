"""Fixed-snapshot diagnostic contrasts; never registered execution routes."""

import triton as tr
import triton.language as tl

from ..periodic_product.kernels import _index, _periodic_raw
from .kernels import _contract


@tr.jit
def profile_cache(
    P,
    Order,
    Cache,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    S: tl.constexpr,
    GROUP: tl.constexpr,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    k = tl.arange(0, S)
    inv = tl.load(P + A + a, valid, 1)
    flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    for side in tl.static_range(2):
        center = tl.load(P + (2 + side) * A + a, valid, 0)
        norm = tl.load(P + (4 + side) * A + a, valid, 1)
        loggrad = tl.load(P + (6 + side) * A + a, valid, 0)
        low = tl.load(P + (9 + side * 2) * A + a, valid, 0).to(tl.int32)
        high = tl.load(P + (10 + side * 2) * A + a, valid, 0).to(tl.int32)
        other_low = tl.load(P + (11 if side == 0 else 9) * A + a, valid, 0)
        other_high = tl.load(P + (12 if side == 0 else 10) * A + a, valid, 0)
        high = tl.where(other_high > other_low, high, low)
        j = low[:, None] + k[None, :]
        if side == 0:
            v, dv = _periodic_raw(j, center[:, None], inv[:, None], OI, LI, NI)
        else:
            v, dv = _periodic_raw(j, center[:, None], inv[:, None], OO, LO, NO)
        safe = tl.min(
            ((norm >= 1e-6) & (norm <= 128.0) & ((flags & (1 << side)) == 0)).to(
                tl.int32
            ),
            0,
        )
        inverse = tl.div_rn(1.0, norm)
        if safe:
            v = v * inverse[:, None]
            dv = dv * inverse[:, None]
        else:
            v = tl.div_rn(v, norm[:, None])
            dv = tl.div_rn(dv, norm[:, None])
        dv = tl.where(
            (flags[:, None] & (1 << side)) != 0, 0.0, dv - loggrad[:, None] * v
        )
        off = (side * 2 * A + pos[:, None]) * S + k[None, :]
        mask = valid[:, None] & (j < high[:, None])
        tl.store(Cache + off, tl.where(mask, v, 0.0), valid[:, None])
        tl.store(Cache + off + A * S, tl.where(mask, dv, 0.0), valid[:, None])


@tr.jit
def cached_contract(
    X,
    P,
    Cache,
    pos,
    a,
    rows,
    valid,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    OUTPUT: tl.constexpr,
    S: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    high = tl.where(other_high > other_low, high, low)
    k = tl.arange(0, BK)
    acc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    grad = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    side: tl.constexpr = 1 if OUTPUT else 0
    for start in range(0, tl.max(tl.maximum(high - low, 0), 0), BK):
        j = low[:, None] + start + k[None, :]
        off = (side * 2 * A + pos[:, None]) * S + start + k[None, :]
        cache_mask = valid[:, None] & ((start + k[None, :]) < S)
        v = tl.load(Cache + off, cache_mask, 0.0)
        dv = tl.load(Cache + off + A * S, cache_mask, 0.0)
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < high[:, None, None])
        )
        xx = tl.load(
            X + rows[None, :, None] * X0 + _index(j, N)[:, None, :] * X1, mask, 0.0
        )
        acc += xx * v[:, None, :]
        grad += xx * dv[:, None, :]
    return tl.sum(acc, 2), tl.sum(grad, 2)


@tr.jit
def fused_probe(
    X,
    DY,
    P,
    Order,
    G,
    Partial,
    Cache,
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
    SORTED_P: tl.constexpr = False,
    SORTED_PARTIAL: tl.constexpr = False,
    CACHED: tl.constexpr = False,
    S: tl.constexpr = 1,
    COMPACT_WRAP: tl.constexpr = False,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    original_a = tl.load(Order + pos, valid, 0).to(tl.int32)
    a = pos if SORTED_P else original_a
    local_rows = tl.program_id(1) * BM + tl.arange(0, BM)
    rows = BSTART + local_rows
    if CACHED:
        g, dg = cached_contract(
            DY, P, Cache, pos, a, rows, valid, A, B, NO, D0, D1, True, S, BM, BK, GROUP
        )
    else:
        g, dg = _contract(
            DY,
            P,
            a,
            rows,
            valid,
            A,
            B,
            NO,
            LO,
            OO,
            D0,
            D1,
            True,
            True,
            BM,
            BK,
            GROUP,
            True,
            COMPACT_WRAP,
        )
    off = ((local_rows[None, :] // 8) * A + pos[:, None]) * 8 + local_rows[None, :] % 8
    tl.store(G + off, g, valid[:, None])
    if CACHED:
        h, dh = cached_contract(
            X, P, Cache, pos, a, rows, valid, A, B, NI, X0, X1, False, S, BM, BK, GROUP
        )
    else:
        h, dh = _contract(
            X,
            P,
            a,
            rows,
            valid,
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
            COMPACT_WRAP,
        )
    dest = pos if SORTED_PARTIAL else original_a
    amp = tl.load(P + a, valid, 0)
    tile = tl.program_id(1)
    tl.store(Partial + tile * 3 * A + dest, tl.sum(h * g, 1), valid)
    tl.store(Partial + tile * 3 * A + A + dest, amp * tl.sum(g * dh, 1), valid)
    tl.store(Partial + tile * 3 * A + 2 * A + dest, amp * tl.sum(h * dg, 1), valid)


@tr.jit
def axis_probe(
    X,
    DY,
    P,
    Order,
    Value,
    Aux,
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
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    MATCHED: tl.constexpr = False,
):
    pos = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = pos < A
    a = tl.load(Order + pos, valid, 0).to(tl.int32)
    rows = BSTART + tl.program_id(1) * BM + tl.arange(0, BM)
    if OUTPUT:
        value, derivative = _contract(
            DY,
            P,
            a,
            rows,
            valid,
            A,
            B,
            NO,
            LO,
            OO,
            D0,
            D1,
            True,
            DERIVATIVE,
            BM,
            BK,
            GROUP,
            True,
        )
    else:
        value, derivative = _contract(
            X,
            P,
            a,
            rows,
            valid,
            A,
            B,
            NI,
            LI,
            OI,
            X0,
            X1,
            False,
            DERIVATIVE,
            BM,
            BK,
            GROUP,
            True,
        )
    off = (tl.program_id(1) * A + pos[:, None]) * BM + tl.arange(0, BM)[None, :]
    tl.store(Value + off, value, valid[:, None])
    if DERIVATIVE:
        tl.store(Aux + off, derivative, valid[:, None])
    elif MATCHED:
        tl.store(Aux + off, value, valid[:, None])
