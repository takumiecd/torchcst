"""Explicit atom/batch lane ownership, with the original local event formulas."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl

from .integer_query_kernels import query as integer_query
from .vector_atomic_kernels import add4


@tr.jit
def _source(Order, slot, A: tl.constexpr):
    if Order is not None:
        return tl.load(Order + slot, slot < A, A).to(tl.int32)
    return slot


@tr.jit
def _point(Pos, a, A: tl.constexpr, DIM: tl.constexpr, EVENT, COMPACT_C2: tl.constexpr):
    point = tl.load(Pos + (DIM * 6 + EVENT) * A + a, a < A, -1).to(tl.int32)
    if COMPACT_C2:
        point = tl.where(point >= 0, point & ((1 << 14) - 1), point)
    return point


@tr.jit
def _coeff(
    Co,
    a,
    A: tl.constexpr,
    DIM: tl.constexpr,
    EVENT,
    LAYOUT: tl.constexpr,
    Pos=None,
    COMPACT_C2: tl.constexpr = False,
):
    tl.static_assert(LAYOUT == "soa")
    if COMPACT_C2:
        c0 = tl.load(Co + (DIM * 13 + EVENT) * A + a, a < A, 0)
        c1 = tl.load(Co + (DIM * 13 + 6 + EVENT) * A + a, a < A, 0)
        q = tl.load(Co + (DIM * 13 + 12) * A + a, a < A, 0)
        raw = tl.load(Pos + (DIM * 6 + EVENT) * A + a, a < A, -1).to(tl.int32)
        knot = tl.where(
            EVENT == 0, 1, tl.where(EVENT == 1, -3, tl.where(EVENT == 2, 3, -1))
        )
        multiplier = tl.where(EVENT < 4, knot, tl.where((raw & (1 << 14)) != 0, -2, 1))
        c2 = q * multiplier
    else:
        c0 = tl.load(Co + ((DIM * 3) * 6 + EVENT) * A + a, a < A, 0)
        c1 = tl.load(Co + ((DIM * 3 + 1) * 6 + EVENT) * A + a, a < A, 0)
        c2 = tl.load(Co + ((DIM * 3 + 2) * 6 + EVENT) * A + a, a < A, 0)
    return c0, c1, c2


@tr.jit
def _pullback(
    Meta, DP, a, total, dout, din, A: tl.constexpr, ADD: tl.constexpr = False
):
    inv = tl.load(Meta + a * 12 + 4, a < A, 0)
    scale = tl.load(Meta + a * 12, a < A, 0) * inv
    for component in tl.static_range(4):
        if component == 0:
            value = total * inv * tl.load(Meta + a * 12 + 7, a < A, 0)
        elif component == 1:
            value = total * inv * tl.load(Meta + a * 12 + 8, a < A, 0)
        elif component == 2:
            value = scale * (dout - total * tl.load(Meta + a * 12 + 5, a < A, 0))
        else:
            value = scale * (din - total * tl.load(Meta + a * 12 + 6, a < A, 0))
        if ADD:
            # Each CTA owns distinct atoms; batch chunks execute on the same stream.
            previous = tl.load(DP + 4 * a + component, a < A, 0)
            tl.store(DP + 4 * a + component, previous + value.to(tl.float32), a < A)
        else:
            tl.store(DP + 4 * a + component, value, a < A)


@tr.jit
def _query_loop(
    Bank,
    Pos,
    Co,
    HalfInv,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    PB: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr,
    COMPACT_C2: tl.constexpr,
):
    value = tl.full((AB, BB), 0, tl.float64, layout=TILE)
    prime = tl.full((AB, BB), 0, tl.float64, layout=TILE)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2)
        term = tl.fma(c2[:, None], p2, tl.fma(c1[:, None], p1, c0[:, None] * p0))
        value += tl.where(event < 4, -1, 1) * term
        if PRIME:
            d0 = -c1 * half
            d1 = -2 * c2 * half
            prime += tl.where(event < 4, -1, 1) * tl.fma(
                d1[:, None], p1, d0[:, None] * p0
            )
    return value, prime


@tr.jit
def _query(
    Bank,
    Pos,
    Co,
    HalfInv,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    PB: tl.constexpr,
    TILE: tl.constexpr,
    LAYOUT: tl.constexpr = "soa",
    INTEGER: tl.constexpr = False,
    Meta=None,
    Origin=None,
    Spacing=None,
    Raw=None,
    Order=None,
    N: tl.constexpr = 0,
    LOOPED: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
):
    if LOOPED:
        tl.static_assert(not INTEGER)
        return _query_loop(
            Bank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            POINTS,
            DIM,
            AB,
            BB,
            PRIME,
            PB,
            TILE,
            LAYOUT,
            COMPACT_C2,
        )
    if INTEGER:
        return integer_query(
            Bank,
            Meta,
            Co,
            Origin,
            Spacing,
            Raw,
            a,
            _source(Order, a, A),
            b,
            A,
            B,
            N,
            PB,
            DIM,
            AB,
            BB,
            PRIME,
            TILE,
        )
    value = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    prime = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in tl.static_range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2)
        v = tl.fma(c2[:, None], p2, tl.fma(c1[:, None], p1, c0[:, None] * p0))
        sign = -1 if event < 4 else 1
        value += sign * v
        if PRIME:
            d0 = -c1 * half
            d1 = -2 * c2 * half
            prime += sign * tl.fma(d1[:, None], p1, d0[:, None] * p0)
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _inject(
    Value,
    Delta,
    Meta,
    Pos,
    Co,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    DIM: tl.constexpr,
    BorderDelta,
    BorderPos,
    SHARDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    Injection,
    ENDPOINTS: tl.constexpr,
    SHARD_ORIGIN: tl.constexpr,
    PB: tl.constexpr,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Program=None,
    VECTOR_RED: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
):
    program = tl.program_id(0) if Program is None else Program
    source = _source(Order, a, A)
    scale = tl.load(Meta + 12 * source, a < A, 0) * tl.load(
        Meta + 12 * source + 4, a < A, 0
    )
    weight = (Value * scale[:, None]).to(Delta.dtype.element_ty)
    for event in tl.static_range(6):
        point = _point(Pos, a, A, DIM, event, COMPACT_C2)
        c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT, Pos, COMPACT_C2)
        if event >= 4:
            point = point + 1
            c0 = tl.fma(4.0, c2, tl.fma(2.0, c1, c0))
            c1 = tl.fma(4.0, c2, c1)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        # Inactive boundary -1 becomes 0 after its +1 shift; coefficients are
        # zero, but retain the original validity for a precise event mask.
        if event >= 4:
            mask &= point[:, None] > 0
        if ENDPOINTS and event < 4:
            code = tl.load(Injection + DIM * A + a, a < A, 0).to(tl.int32)
            mode = (code >> (2 * event)) & 3
            mask &= mode[:, None] != 0
            if SHARD_ORIGIN:
                origin_block = BLOCKS + 1 - ((BLOCKS + 1) >> (code >> 8))
        for p in tl.static_range(3):
            c = c0 if p == 0 else c1 if p == 1 else c2
            if SHARDS and event >= 4:
                block = tl.load(
                    BorderPos + (DIM * 2 + event - 4) * A + a, a < A, -1
                ).to(tl.int32)
                shard = program % SHARDS
                addr = (
                    p * BLOCKS * SHARDS * PB
                    + (block[:, None] * SHARDS + shard) * PB
                    + b[None, :]
                )
                if VECTOR_RED:
                    add4(
                        BorderDelta + addr,
                        weight * c[:, None],
                        mask & (block[:, None] >= 0),
                    )
                else:
                    tl.atomic_add(
                        BorderDelta + addr,
                        weight * c[:, None],
                        mask & (block[:, None] >= 0),
                        sem="relaxed",
                    )
            else:
                addr = Delta + p * POINTS * PB + point[:, None] * PB + b[None, :]
                if ENDPOINTS and SHARD_ORIGIN:
                    shard = program % SHARDS
                    border = (
                        BorderDelta
                        + p * BLOCKS * SHARDS * PB
                        + (origin_block[:, None] * SHARDS + shard) * PB
                        + b[None, :]
                    )
                    addr = tl.where(mode[:, None] == 2, border, addr)
                if VECTOR_RED:
                    add4(addr, weight * c[:, None], mask)
                else:
                    tl.atomic_add(addr, weight * c[:, None], mask, sem="relaxed")


@tr.jit
def forward(
    Bank,
    Meta,
    Pos,
    Co,
    HalfInv,
    H,
    HP,
    Delta,
    Count,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PO: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SAVE: tl.constexpr,
    BorderDelta=None,
    BorderPos=None,
    SHARDS: tl.constexpr = 0,
    BLOCKS: tl.constexpr = 0,
    Injection=None,
    ENDPOINTS: tl.constexpr = False,
    SHARD_ORIGIN: tl.constexpr = False,
    PAD: tl.constexpr = 0,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Base=None,
    LANE_ATOMS: tl.constexpr = 4,
    BATCH_VALUES: tl.constexpr = 1,
    VECTOR_RED: tl.constexpr = False,
    INTEGER: tl.constexpr = False,
    QueryOrigin=None,
    QuerySpacing=None,
    RawInput=None,
    RawAdjoint=None,
    I_NO: tl.constexpr = 0,
    I_NI: tl.constexpr = 0,
    LOOPED_QUERY: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        tl.static_assert(
            BATCH_VALUES == 1 or (AB >= 16 and BB >= 32 and LANE_ATOMS == 4)
        )
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, BATCH_VALUES],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        a = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        h, hp = _query(
            Bank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            PI,
            1,
            AB,
            BB,
            SAVE == "hp",
            B + PAD,
            tile,
            LAYOUT,
            INTEGER=INTEGER,
            Meta=Meta,
            Origin=QueryOrigin,
            Spacing=QuerySpacing,
            Raw=RawInput,
            Order=Order,
            N=I_NI,
            LOOPED=LOOPED_QUERY,
            COMPACT_C2=COMPACT_C2,
        )
        mask = (a[:, None] < A) & (b[None, :] < B)
        if SAVE != "none":
            tl.store(H + a[:, None] * B + b[None, :], h.to(tl.float32), mask)
        if SAVE == "hp":
            tl.store(HP + a[:, None] * B + b[None, :], hp.to(tl.float32), mask)
        _inject(
            h,
            Delta,
            Meta,
            Pos,
            Co,
            a,
            b,
            A,
            B,
            PO,
            0,
            BorderDelta,
            BorderPos,
            SHARDS,
            BLOCKS,
            Injection,
            ENDPOINTS,
            SHARD_ORIGIN,
            B + PAD,
            Order,
            LAYOUT,
            Program=None if Base is None else (base - tl.load(Count)) // AB,
            VECTOR_RED=VECTOR_RED,
            COMPACT_C2=COMPACT_C2,
        )


@tr.jit
def backward(
    InputBank,
    AdjointBank,
    Meta,
    Pos,
    Co,
    HalfInv,
    H,
    HP,
    Delta,
    DP,
    Count,
    A: tl.constexpr,
    B: tl.constexpr,
    PI: tl.constexpr,
    PO: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SAVE: tl.constexpr,
    DX: tl.constexpr,
    VJP: tl.constexpr,
    CONTRACT32: tl.constexpr,
    BorderDelta=None,
    BorderPos=None,
    SHARDS: tl.constexpr = 0,
    BLOCKS: tl.constexpr = 0,
    Injection=None,
    ENDPOINTS: tl.constexpr = False,
    SHARD_ORIGIN: tl.constexpr = False,
    PAD: tl.constexpr = 0,
    ADD_DP: tl.constexpr = False,
    Order=None,
    LAYOUT: tl.constexpr = "soa",
    Base=None,
    LANE_ATOMS: tl.constexpr = 4,
    BATCH_VALUES: tl.constexpr = 1,
    VECTOR_RED: tl.constexpr = False,
    INTEGER: tl.constexpr = False,
    QueryOrigin=None,
    QuerySpacing=None,
    RawInput=None,
    RawAdjoint=None,
    I_NO: tl.constexpr = 0,
    I_NI: tl.constexpr = 0,
    LOOPED_QUERY: tl.constexpr = False,
    COMPACT_C2: tl.constexpr = False,
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        tl.static_assert(
            BATCH_VALUES == 1 or (AB >= 16 and BB >= 32 and LANE_ATOMS == 4)
        )
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, BATCH_VALUES],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        a = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        g, gp = _query(
            AdjointBank,
            Pos,
            Co,
            HalfInv,
            a,
            b,
            A,
            B,
            PO,
            0,
            AB,
            BB,
            VJP,
            B + PAD,
            tile,
            LAYOUT,
            INTEGER=INTEGER,
            Meta=Meta,
            Origin=QueryOrigin,
            Spacing=QuerySpacing,
            Raw=RawAdjoint,
            Order=Order,
            N=I_NO,
            LOOPED=LOOPED_QUERY,
            COMPACT_C2=COMPACT_C2,
        )
        if VJP:
            mask = (a[:, None] < A) & (b[None, :] < B)
            if SAVE == "hp":
                h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
                hp = tl.load(HP + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            else:
                h, hp = _query(
                    InputBank,
                    Pos,
                    Co,
                    HalfInv,
                    a,
                    b,
                    A,
                    B,
                    PI,
                    1,
                    AB,
                    BB,
                    True,
                    B + PAD,
                    tile,
                    LAYOUT,
                    INTEGER=INTEGER,
                    Meta=Meta,
                    Origin=QueryOrigin,
                    Spacing=QuerySpacing,
                    Raw=RawInput,
                    Order=Order,
                    N=I_NI,
                    LOOPED=LOOPED_QUERY,
                    COMPACT_C2=COMPACT_C2,
                )
                if SAVE == "h":
                    h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            if CONTRACT32:
                h, g, hp, gp = (
                    h.to(tl.float32),
                    g.to(tl.float32),
                    hp.to(tl.float32),
                    gp.to(tl.float32),
                )
            _pullback(
                Meta,
                DP,
                _source(Order, a, A),
                tl.sum(h * g, 1).to(tl.float64),
                tl.sum(h * gp, 1).to(tl.float64),
                tl.sum(hp * g, 1).to(tl.float64),
                A,
                ADD_DP,
            )
        if DX:
            _inject(
                g,
                Delta,
                Meta,
                Pos,
                Co,
                a,
                b,
                A,
                B,
                PI,
                1,
                BorderDelta,
                BorderPos,
                SHARDS,
                BLOCKS,
                Injection,
                ENDPOINTS,
                SHARD_ORIGIN,
                B + PAD,
                Order,
                LAYOUT,
                Program=None if Base is None else (base - tl.load(Count)) // AB,
                VECTOR_RED=VECTOR_RED,
                COMPACT_C2=COMPACT_C2,
            )
