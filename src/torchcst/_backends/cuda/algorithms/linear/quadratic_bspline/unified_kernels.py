"""Fixed-slot native/moment execution; scratch sums retain FP64 destinations."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl

from .gluon_kernels import _inject, _pullback, _query, _source


@tr.jit
def _term(
    X,
    Co,
    slot,
    a,
    b,
    left,
    right,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    SITE: tl.constexpr,
    PRIME: tl.constexpr,
    POINT_MAJOR: tl.constexpr = False,
):
    mask = (
        (a[:, None] < A) & (b[None, :] < B) & ((left + SITE)[:, None] < right[:, None])
    )
    addr = (
        (left + SITE)[:, None] * B + b[None, :]
        if POINT_MAJOR
        else b[None, :] * N + (left + SITE)[:, None]
    )
    x = tl.load(X + addr, mask, 0)
    f = tl.load(Co + (DIM * 18 + SITE) * A + slot, a < A, 0)
    v = x * f[:, None]
    if PRIME:
        df = tl.load(Co + (DIM * 18 + 8 + SITE) * A + slot, a < A, 0)
        p = x * df[:, None]
    else:
        p = tl.full(v.shape, 0, tl.float32, layout=v.type.layout)
    return v, p


@tr.jit
def _native_query(
    X,
    Co,
    slot,
    a,
    b,
    left,
    right,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    UNROLL: tl.constexpr,
    POINT_MAJOR: tl.constexpr = False,
):
    v0, p0 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 0, PRIME, POINT_MAJOR)
    v1, p1 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 1, PRIME, POINT_MAJOR)
    s0, t0 = v0 + v1, p0 + p1
    v2, p2 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 2, PRIME, POINT_MAJOR)
    v3, p3 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 3, PRIME, POINT_MAJOR)
    s1, t1 = v2 + v3, p2 + p3
    v4, p4 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 4, PRIME, POINT_MAJOR)
    v5, p5 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 5, PRIME, POINT_MAJOR)
    s2, t2 = v4 + v5, p4 + p5
    v6, p6 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 6, PRIME, POINT_MAJOR)
    v7, p7 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 7, PRIME, POINT_MAJOR)
    value, prime = (s0 + s1) + (s2 + (v6 + v7)), (t0 + t1) + (t2 + (p6 + p7))
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _native_inject(
    Value,
    Y,
    Meta,
    Co,
    slot,
    a,
    b,
    left,
    right,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    UNROLL: tl.constexpr,
    SHARDS: tl.constexpr = 0,
    STRIDE: tl.constexpr = 0,
    POINT_MAJOR: tl.constexpr = False,
    ROWS: tl.constexpr = 0,
):
    scale = tl.load(Meta + 12 * a, a < A, 0) * tl.load(Meta + 12 * a + 4, a < A, 0)
    weight = (Value * scale[:, None]).to(tl.float32)
    offset = (tl.program_id(0) % SHARDS) * STRIDE if SHARDS else 0
    for site in tl.static_range(8):
        f = tl.load(Co + (DIM * 18 + site) * A + slot, a < A, 0)
        j = left + site
        mask = (a[:, None] < A) & (b[None, :] < B) & (j[:, None] < right[:, None])
        addr = (
            j[:, None] * ROWS + b[None, :]
            if POINT_MAJOR
            else b[None, :] * N + j[:, None]
        )
        tl.atomic_add(
            Y + offset + addr,
            weight * f[:, None],
            mask,
            sem="relaxed",
        )


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
    LANE_ATOMS: tl.constexpr = 1,
    RawX=None,
    RawDY=None,
    NativeOut=None,
    NO: tl.constexpr = 0,
    NI: tl.constexpr = 0,
    NativeHotOut=None,
    HOT_SHARDS: tl.constexpr = 0,
    HOT_STRIDE: tl.constexpr = 0,
    NATIVE_POINT_MAJOR: tl.constexpr = False,
    HOT_ROWS: tl.constexpr = 0,
):
    tl.static_assert(SAVE == "none")
    count = tl.load(Count)
    base = tl.program_id(0) * AB
    if base < A:
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, 1],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        slot = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        a = tl.where(slot >= count, slot, A)
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        if base < count:
            kind = tl.load(Co + 16 * A + base).to(tl.int32)
            if kind == 1:
                ns = tl.where(slot < count, slot, A)
                na = _source(Order, ns, A)
                li = tl.load(Pos + 6 * A + ns, ns < A, 0)
                ri = tl.load(Pos + 7 * A + ns, ns < A, 0)
                lo = tl.load(Pos + ns, ns < A, 0)
                ro = tl.load(Pos + A + ns, ns < A, 0)
                nh, _ = _native_query(
                    RawX,
                    Co,
                    ns,
                    na,
                    b,
                    li,
                    ri,
                    A,
                    B,
                    NI,
                    1,
                    AB,
                    BB,
                    False,
                    True,
                    NATIVE_POINT_MAJOR,
                )
                _native_inject(
                    nh,
                    NativeHotOut if HOT_SHARDS else NativeOut,
                    Meta,
                    Co,
                    ns,
                    na,
                    b,
                    lo,
                    ro,
                    A,
                    B,
                    NO,
                    0,
                    True,
                    HOT_SHARDS,
                    HOT_STRIDE,
                    NATIVE_POINT_MAJOR,
                    HOT_ROWS,
                )
        if base + AB > count:
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
                Program=tl.program_id(0),
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
    LANE_ATOMS: tl.constexpr = 1,
    RawX=None,
    RawDY=None,
    NativeOut=None,
    NO: tl.constexpr = 0,
    NI: tl.constexpr = 0,
    NativeHotOut=None,
    HOT_SHARDS: tl.constexpr = 0,
    HOT_STRIDE: tl.constexpr = 0,
    NATIVE_POINT_MAJOR: tl.constexpr = False,
    HOT_ROWS: tl.constexpr = 0,
):
    tl.static_assert(SAVE == "none")
    count = tl.load(Count)
    base = tl.program_id(0) * AB
    if base < A:
        atom_warps: tl.constexpr = min(AB // LANE_ATOMS, 4)
        tile: tl.constexpr = tl.BlockedLayout(
            [1, 1],
            [LANE_ATOMS, 32 // LANE_ATOMS],
            [atom_warps, 4 // atom_warps],
            [1, 0],
        )
        slot = base + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
        a = tl.where(slot >= count, slot, A)
        b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
        if base < count:
            kind = tl.load(Co + 16 * A + base).to(tl.int32)
            if kind == 1:
                ns = tl.where(slot < count, slot, A)
                na = _source(Order, ns, A)
                li = tl.load(Pos + 6 * A + ns, ns < A, 0)
                ri = tl.load(Pos + 7 * A + ns, ns < A, 0)
                lo = tl.load(Pos + ns, ns < A, 0)
                ro = tl.load(Pos + A + ns, ns < A, 0)
                ng, ngp = _native_query(
                    RawDY,
                    Co,
                    ns,
                    na,
                    b,
                    lo,
                    ro,
                    A,
                    B,
                    NO,
                    0,
                    AB,
                    BB,
                    VJP,
                    True,
                    NATIVE_POINT_MAJOR,
                )
                if VJP:
                    nh, nhp = _native_query(
                        RawX,
                        Co,
                        ns,
                        na,
                        b,
                        li,
                        ri,
                        A,
                        B,
                        NI,
                        1,
                        AB,
                        BB,
                        True,
                        True,
                        NATIVE_POINT_MAJOR,
                    )
                    if CONTRACT32:
                        nh, ng, nhp, ngp = (
                            nh.to(tl.float32),
                            ng.to(tl.float32),
                            nhp.to(tl.float32),
                            ngp.to(tl.float32),
                        )
                    _pullback(
                        Meta,
                        DP,
                        na,
                        tl.sum(nh * ng, 1).to(tl.float64),
                        tl.sum(nh * ngp, 1).to(tl.float64),
                        tl.sum(nhp * ng, 1).to(tl.float64),
                        A,
                        ADD_DP,
                    )
                if DX:
                    _native_inject(
                        ng,
                        NativeHotOut if HOT_SHARDS else NativeOut,
                        Meta,
                        Co,
                        ns,
                        na,
                        b,
                        li,
                        ri,
                        A,
                        B,
                        NI,
                        1,
                        True,
                        HOT_SHARDS,
                        HOT_STRIDE,
                        NATIVE_POINT_MAJOR,
                        HOT_ROWS,
                    )
        if base + AB > count:
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
                    hp = tl.load(HP + a[:, None] * B + b[None, :], mask, 0).to(
                        tl.float64
                    )
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
                        h = tl.load(H + a[:, None] * B + b[None, :], mask, 0).to(
                            tl.float64
                        )
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
                    Program=tl.program_id(0),
                )
