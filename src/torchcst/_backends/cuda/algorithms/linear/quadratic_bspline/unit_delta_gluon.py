"""Float-pair range queries, balanced integer injection, and source gradients."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl


@tr.jit
def _two_sum(a, b):
    s = a + b
    v = s - a
    return s, (a - (s - v)) + (b - v)


@tr.jit
def _add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    return _two_sum(s, e + (al + bl))


@tr.jit
def _mul(ah, al, bh, bl):
    p = ah * bh
    e = tl.fma(ah, bh, -p)
    e += ah * bl
    e += al * bh
    e += al * bl
    return _two_sum(p, e)


@tr.jit
def _scale(power):
    return ((1023 + power.to(tl.int64)) << 52).to(tl.float64, bitcast=True)


@tr.jit
def _retain64(value):
    # Preserve the float-pair evaluation through consumers that immediately cast
    # to FP32. A 64-bit bitwise identity does not change the numerical result.
    return tl.inline_asm_elementwise(
        "mov.b64 $0, $1;", "=d,d", [value], tl.float64, True, 1
    )


@tr.jit
def _query(
    Bank,
    Co,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    PB: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    PRIME: tl.constexpr,
    TILE: tl.constexpr,
):
    cuts = (Co + 40 * A).to(tl.pointer_type(tl.int32))
    native = tl.load(cuts + a, a < A, -1) < 0
    vh = tl.full((AB, BB), 0.0, tl.float32, layout=TILE)
    vl = tl.full((AB, BB), 0.0, tl.float32, layout=TILE)
    ph = tl.full((AB, BB), 0.0, tl.float32, layout=TILE)
    pl = tl.full((AB, BB), 0.0, tl.float32, layout=TILE)
    for k in tl.static_range(4):
        cut = tl.load(cuts + (DIM * 4 + k) * A + a, a < A, 0)
        for j in tl.static_range(3):
            c = tl.load(Co + (DIM * 20 + k * 3 + j) * A + a, (a < A) & ~native, 0)
            c64 = c.to(tl.float64) * 2.220446049250313e-16
            ch = c64.to(tl.float32)
            cl = (c64 - ch.to(tl.float64)).to(tl.float32)
            mask = (
                (a[:, None] < A)
                & ~native[:, None]
                & (b[None, :] < B)
                & ((cut[:, None] + j) < N)
            )
            bh = tl.load(
                Bank + (2 * (N + 1) + cut[:, None] + j) * PB + b[None, :], mask, 0
            )
            bl = tl.load(
                Bank + (3 * (N + 1) + cut[:, None] + j) * PB + b[None, :], mask, 0
            )
            mh, ml = _mul(bh, bl, ch[:, None], cl[:, None])
            vh, vl = _add(vh, vl, mh, ml)
            if PRIME and j < 2:
                c = tl.load(
                    Co + (DIM * 20 + 12 + k * 2 + j) * A + a, (a < A) & ~native, 0
                )
                c64 = c.to(tl.float64) * 2.220446049250313e-16
                ch = c64.to(tl.float32)
                cl = (c64 - ch.to(tl.float64)).to(tl.float32)
                bh = tl.load(Bank + (cut[:, None] + j) * PB + b[None, :], mask, 0)
                bl = tl.load(
                    Bank + ((N + 1) + cut[:, None] + j) * PB + b[None, :], mask, 0
                )
                mh, ml = _mul(bh, bl, ch[:, None], cl[:, None])
                ph, pl = _add(ph, pl, mh, ml)
    xp = tl.load(Bank + N * PB + b, b < B, 0).to(tl.int32)
    scale = _scale(-xp)
    value = (vh.to(tl.float64) + vl.to(tl.float64)) * scale[None, :]
    prime = (
        (ph.to(tl.float64) + pl.to(tl.float64))
        * scale[None, :]
        / tl.load(Spacing + DIM)
        if PRIME
        else tl.full((AB, BB), 0.0, tl.float64, layout=TILE)
    )
    return _retain64(value), _retain64(prime)


@tr.jit
def _inject(
    Value,
    Meta,
    Co,
    Delta,
    Bank,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    RAWN: tl.constexpr,
    PB: tl.constexpr,
    DIM: tl.constexpr,
    SHARDS: tl.constexpr,
):
    cuts = (Co + 40 * A).to(tl.pointer_type(tl.int32))
    native = tl.load(cuts + a, a < A, -1) < 0
    no = tl.load(Meta + 12 * A + a, a < A, 0.0)
    ni = tl.load(Meta + 13 * A + a, a < A, 0.0)
    lam = (
        tl.load(Meta + 12 * a, a < A, 0.0)
        * no
        * ni
        * tl.load(Meta + 12 * a + 4, a < A, 0.0)
    )
    xp = tl.load(Bank + RAWN * PB + b, b < B, 0).to(tl.int32)
    wp = tl.load(Bank + (2 * RAWN + 1) * PB + b, b < B, 0).to(tl.int32)
    weighted = Value * lam[:, None] * _scale(xp + wp)[None, :]
    wi = tl.inline_asm_elementwise(
        "cvt.rni.s64.f64 $0, $1;", "=l,d", [weighted], tl.int64, True, 1
    )
    shard = tl.program_id(0) % SHARDS
    for k in tl.static_range(4):
        cut = tl.load(cuts + (DIM * 4 + k) * A + a, a < A, 0)
        for j in tl.static_range(3):
            c = tl.load(Co + (DIM * 20 + k * 3 + j) * A + a, (a < A) & ~native, 0)
            ph = tl.inline_asm_elementwise(
                "mul.hi.s64 $0, $1, $2;", "=l,l,l", [wi, c[:, None]], tl.int64, True, 1
            )
            pl = wi.to(tl.uint64) * c[:, None].to(tl.uint64)
            low = pl & 281474976710655
            high = ((ph.to(tl.uint64) << 16) | (pl >> 48)).to(tl.int64)
            carry = low >= 140737488355328
            low = (low - tl.where(carry, 281474976710656, 0).to(tl.uint64)).to(tl.int64)
            high += carry.to(tl.int64)
            mask = (
                (a[:, None] < A)
                & ~native[:, None]
                & (b[None, :] < B)
                & ((cut[:, None] + j) < N)
                & (c[:, None] != 0)
                & (wi != 0)
            )
            addr = (shard * N + cut[:, None] + j) * PB + b[None, :]
            tl.atomic_add(Delta + addr, low, mask, sem="relaxed")
            tl.atomic_add(Delta + SHARDS * N * PB + addr, high, mask, sem="relaxed")


@tr.jit
def forward(
    Bank,
    Meta,
    Co,
    Spacing,
    Delta,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SHARDS: tl.constexpr,
    LANES: tl.constexpr = 1,
):
    aw: tl.constexpr = min(AB // LANES, 4)
    tile: tl.constexpr = tl.BlockedLayout(
        [1, 1], [LANES, 32 // LANES], [aw, 4 // aw], [1, 0]
    )
    a = tl.program_id(0) * AB + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
    b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
    h, _ = _query(Bank, Co, Spacing, a, b, A, B, NI, PB, 1, AB, BB, False, tile)
    _inject(h, Meta, Co, Delta, Bank, a, b, A, B, NO, NI, PB, 0, SHARDS)


@tr.jit
def backward(
    Input,
    Adjoint,
    Meta,
    Co,
    Spacing,
    Delta,
    DP,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    PB: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    SHARDS: tl.constexpr,
    DX: tl.constexpr,
    VJP: tl.constexpr,
    ADD: tl.constexpr,
    CONTRACT32: tl.constexpr,
    LANES: tl.constexpr = 1,
):
    aw: tl.constexpr = min(AB // LANES, 4)
    tile: tl.constexpr = tl.BlockedLayout(
        [1, 1], [LANES, 32 // LANES], [aw, 4 // aw], [1, 0]
    )
    a = tl.program_id(0) * AB + tl.arange(0, AB, layout=tl.SliceLayout(1, tile))
    b = tl.arange(0, BB, layout=tl.SliceLayout(0, tile))
    g, gp = _query(Adjoint, Co, Spacing, a, b, A, B, NO, PB, 0, AB, BB, VJP, tile)
    if DX:
        _inject(g, Meta, Co, Delta, Adjoint, a, b, A, B, NI, NO, PB, 1, SHARDS)
    if VJP:
        h, hp = _query(Input, Co, Spacing, a, b, A, B, NI, PB, 1, AB, BB, True, tile)
        if CONTRACT32:
            h, g, hp, gp = (
                h.to(tl.float32),
                g.to(tl.float32),
                hp.to(tl.float32),
                gp.to(tl.float32),
            )
        total = tl.sum(h * g, 1).to(tl.float64)
        dout = tl.sum(h * gp, 1).to(tl.float64)
        din = tl.sum(hp * g, 1).to(tl.float64)
        ratio = (
            tl.load(Meta + 12 * A + a, a < A, 0.0)
            * tl.load(Meta + 13 * A + a, a < A, 0.0)
            * tl.load(Meta + 12 * a + 4, a < A, 0.0)
        )
        amp = tl.load(Meta + 12 * a, a < A, 0.0)
        cuts = (Co + 40 * A).to(tl.pointer_type(tl.int32))
        live = (a < A) & (tl.load(cuts + a, a < A, -1) >= 0)
        for k in tl.static_range(4):
            if k == 0:
                val = ratio * total * tl.load(Meta + 12 * a + 7, a < A, 0.0)
            elif k == 1:
                val = ratio * total * tl.load(Meta + 12 * a + 8, a < A, 0.0)
            elif k == 2:
                val = (
                    amp
                    * ratio
                    * (dout - total * tl.load(Meta + 12 * a + 5, a < A, 0.0))
                )
            else:
                val = (
                    amp * ratio * (din - total * tl.load(Meta + 12 * a + 6, a < A, 0.0))
                )
            if ADD:
                val = tl.load(DP + 4 * a + k, live, 0) + val.to(tl.float32)
            tl.store(DP + 4 * a + k, val, live)
