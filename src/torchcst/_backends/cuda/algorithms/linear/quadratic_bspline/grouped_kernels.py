"""Shared input prefix reads and a grouped adjoint; unchanged source pullback."""

import triton as tr
import triton.language as tl

from . import local_kernels as lk
from .local_kernels import _coeff, _source
from .parallel_kernels import _pullback


@tr.jit
def _members(Ends, IDs, ChunkEnds, Mapping, chunk, A: tl.constexpr, AB: tl.constexpr):
    bucket = tl.load(Mapping + chunk)
    begin = tl.load(Ends + bucket - 1, bucket > 0, 0)
    end = tl.load(Ends + bucket)
    first_chunk = tl.load(ChunkEnds + bucket - 1, bucket > 0, 0)
    start = begin + (chunk - first_chunk) * AB
    index = start + tl.arange(0, AB)
    a = tl.load(IDs + index, index < end, A)
    first = tl.load(IDs + start)
    return a, first


@tr.jit
def _shared(
    Bank,
    Pos,
    Co,
    HalfInv,
    a,
    first,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    PB: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    value = tl.full((AB, BB), 0, tl.float32)
    prime = tl.full((AB, BB), 0, tl.float32)
    half = tl.load(HalfInv + A + a, a < A, 0)
    for event in tl.static_range(6):
        point = tl.load(Pos + (6 + event) * A + first)
        mask = (b < B) & (point >= 0)
        p0 = tl.load(Bank + point * PB + b, mask, 0)
        p1 = tl.load(Bank + POINTS * PB + point * PB + b, mask, 0)
        p2 = tl.load(Bank + 2 * POINTS * PB + point * PB + b, mask, 0)
        c0, c1, c2 = _coeff(Co, a, A, 1, event, LAYOUT)
        v = tl.fma(
            c2[:, None],
            p2[None, :],
            tl.fma(c1[:, None], p1[None, :], c0[:, None] * p0[None, :]),
        )
        sign = -1 if event < 4 else 1
        value += sign * v
        if PRIME:
            d0, d1 = -c1 * half, -2 * c2 * half
            prime += sign * tl.fma(d1[:, None], p1[None, :], d0[:, None] * p0[None, :])
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _reduced(
    Value,
    Delta,
    Meta,
    Pos,
    Co,
    a,
    first,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    POINTS: tl.constexpr,
    BorderDelta,
    BorderPos,
    SHARDS: tl.constexpr,
    BLOCKS: tl.constexpr,
    Injection,
    ENDPOINTS: tl.constexpr,
    SHARD_ORIGIN: tl.constexpr,
    PB: tl.constexpr,
    Order,
    LAYOUT: tl.constexpr,
    Program,
):
    source = _source(Order, a, A)
    scale = tl.load(Meta + 12 * source, a < A, 0) * tl.load(
        Meta + 12 * source + 4, a < A, 0
    )
    weight = (Value * scale[:, None]).to(tl.float32)
    for event in tl.static_range(6):
        point = tl.load(Pos + (6 + event) * A + first)
        c0, c1, c2 = _coeff(Co, a, A, 1, event, LAYOUT)
        if event >= 4:
            point += 1
            c0 = tl.fma(4.0, c2, tl.fma(2.0, c1, c0))
            c1 = tl.fma(4.0, c2, c1)
        mask = (b < B) & (point >= 0)
        if event >= 4:
            mask &= point > 0
        if ENDPOINTS and event < 4:
            code = tl.load(Injection + A + first).to(tl.int32)
            mode = (code >> (2 * event)) & 3
            mask &= mode != 0
            if SHARD_ORIGIN:
                origin_block = BLOCKS + 1 - ((BLOCKS + 1) >> (code >> 8))
        for p in tl.static_range(3):
            c = c0 if p == 0 else c1 if p == 1 else c2
            value = tl.sum(weight * c[:, None], 0)
            shard = Program % SHARDS if SHARDS else 0
            if SHARDS and event >= 4:
                block = tl.load(BorderPos + (2 + event - 4) * A + first)
                addr = (
                    BorderDelta
                    + p * BLOCKS * SHARDS * PB
                    + (block * SHARDS + shard) * PB
                    + b
                )
                tl.atomic_add(addr, value, mask & (block >= 0), sem="relaxed")
            else:
                addr = Delta + p * POINTS * PB + point * PB + b
                if ENDPOINTS and SHARD_ORIGIN and event < 4:
                    border = (
                        BorderDelta
                        + p * BLOCKS * SHARDS * PB
                        + (origin_block * SHARDS + shard) * PB
                        + b
                    )
                    addr = tl.where(mode == 2, border, addr)
                tl.atomic_add(addr, value, mask, sem="relaxed")


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
    GroupEnds=None,
    GroupIDs=None,
    ChunkEnds=None,
    ChunkMap=None,
    CAP: tl.constexpr = 0,
    SHARED_INPUT: tl.constexpr = True,
    REDUCE_DX: tl.constexpr = True,
):
    for chunk in range(
        tl.program_id(0), tl.load(ChunkEnds + CAP - 1), tl.num_programs(0)
    ):
        a, first = _members(GroupEnds, GroupIDs, ChunkEnds, ChunkMap, chunk, A, AB)
        b = tl.arange(0, BB)
        if SHARED_INPUT:
            h, hp = _shared(
                Bank,
                Pos,
                Co,
                HalfInv,
                a,
                first,
                b,
                A,
                B,
                PI,
                AB,
                BB,
                SAVE == "hp",
                B + PAD,
                LAYOUT,
            )
        else:
            h, hp = lk._query(
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
                LAYOUT,
            )
        mask = (a[:, None] < A) & (b[None, :] < B)
        if SAVE != "none":
            tl.store(H + a[:, None] * B + b[None, :], h.to(tl.float32), mask)
        if SAVE == "hp":
            tl.store(HP + a[:, None] * B + b[None, :], hp.to(tl.float32), mask)
        lk._inject(
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
            Program=chunk,
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
    GroupEnds=None,
    GroupIDs=None,
    ChunkEnds=None,
    ChunkMap=None,
    CAP: tl.constexpr = 0,
    SHARED_INPUT: tl.constexpr = True,
    REDUCE_DX: tl.constexpr = True,
):
    for chunk in range(
        tl.program_id(0), tl.load(ChunkEnds + CAP - 1), tl.num_programs(0)
    ):
        a, first = _members(GroupEnds, GroupIDs, ChunkEnds, ChunkMap, chunk, A, AB)
        b = tl.arange(0, BB)
        g, gp = lk._query(
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
            LAYOUT,
        )
        if VJP:
            mask = (a[:, None] < A) & (b[None, :] < B)
            if SAVE == "hp":
                h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
                hp = tl.load(HP + a[:, None] * B + b[None, :], mask, 0).to(tl.float64)
            else:
                if SHARED_INPUT:
                    h, hp = _shared(
                        InputBank,
                        Pos,
                        Co,
                        HalfInv,
                        a,
                        first,
                        b,
                        A,
                        B,
                        PI,
                        AB,
                        BB,
                        True,
                        B + PAD,
                        LAYOUT,
                    )
                else:
                    h, hp = lk._query(
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
            if REDUCE_DX:
                _reduced(
                    g,
                    Delta,
                    Meta,
                    Pos,
                    Co,
                    a,
                    first,
                    b,
                    A,
                    B,
                    PI,
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
                    chunk,
                )
            else:
                lk._inject(
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
                    Program=chunk,
                )
