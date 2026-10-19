"""Explicit atom/batch lane ownership, with the original local event formulas."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl


@tr.jit
def _source(Order, slot, A: tl.constexpr):
    if Order is not None:
        return tl.load(Order + slot, slot < A, A).to(tl.int32)
    return slot


@tr.jit
def _coeff(
    Co, a, A: tl.constexpr, DIM: tl.constexpr, EVENT: tl.constexpr, LAYOUT: tl.constexpr
):
    tl.static_assert(LAYOUT == "soa")
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
):
    value = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    prime = tl.full((AB, BB), 0, tl.float32, layout=TILE)
    half = tl.load(HalfInv + DIM * A + a, a < A, 0)
    for event in tl.static_range(6):
        point = tl.load(Pos + (DIM * 6 + event) * A + a, a < A, -1)
        mask = (a[:, None] < A) & (b[None, :] < B) & (point[:, None] >= 0)
        p0 = tl.load(Bank + point[:, None] * PB + b[None, :], mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point[:, None] * PB + b[None, :], mask, 0)
        c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT)
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
):
    program = tl.program_id(0) if Program is None else Program
    source = _source(Order, a, A)
    scale = tl.load(Meta + 12 * source, a < A, 0) * tl.load(
        Meta + 12 * source + 4, a < A, 0
    )
    weight = (Value * scale[:, None]).to(tl.float32)
    for event in tl.static_range(6):
        point = tl.load(Pos + (DIM * 6 + event) * A + a, a < A, -1)
        c0, c1, c2 = _coeff(Co, a, A, DIM, event, LAYOUT)
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
                block = tl.load(BorderPos + (DIM * 2 + event - 4) * A + a, a < A, -1)
                shard = program % SHARDS
                addr = (
                    p * BLOCKS * SHARDS * PB
                    + (block[:, None] * SHARDS + shard) * PB
                    + b[None, :]
                )
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
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, 1],
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
):
    base = tl.load(Count) + tl.program_id(0) * AB if Base is None else Base
    if base < A:
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, 1],
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
            )
