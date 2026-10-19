"""Per-forward native profiles in disjoint native Co/Pos slots; exact fallbacks."""

import triton as tr
import triton.language as tl

from . import native_kernels as nk
from . import native_mixed_kernels as mk
from .kernels import _shape
from .local_kernels import _source
from .parallel_kernels import _pullback


@tr.jit
def prepare(
    Meta,
    Pos,
    Co,
    Count,
    Order,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    base = tl.program_id(0) * 128
    count = tl.load(Count)
    if base < count:
        slot = base + tl.arange(0, 128)
        valid = slot < count
        source = _source(Order, tl.where(valid, slot, A), A)
        for dim in tl.static_range(2):
            n = NO if dim == 0 else NI
            radius, ir, center, origin, spacing, left, right = nk._axis(
                Meta, Origin, Spacing, source, A, n, dim, RECIPROCAL
            )
            tl.store(Pos + (dim * 6) * A + slot, left, valid)
            tl.store(Pos + (dim * 6 + 1) * A + slot, right, valid)
            distance = (origin + left.to(tl.float64) * spacing - center).to(tl.float32)
            for site in tl.static_range(8):
                d = distance + site * spacing.to(tl.float32)
                f, df = _shape(d, radius.to(tl.float32), ir.to(tl.float32), RECIPROCAL)
                tl.store(Co + (dim * 18 + site) * A + slot, f, valid)
                tl.store(Co + (dim * 18 + 8 + site) * A + slot, df, valid)


@tr.jit
def _bounds(Pos, slot, count, A: tl.constexpr):
    valid = slot < count
    lo = tl.load(Pos + slot, valid, 0)
    ro = tl.load(Pos + A + slot, valid, 0)
    li = tl.load(Pos + 6 * A + slot, valid, 0)
    ri = tl.load(Pos + 7 * A + slot, valid, 0)
    safe = tl.min((~valid | (tl.minimum(ro - lo, ri - li) > 2)).to(tl.int32), 0) > 0
    fit = tl.min((~valid | (tl.maximum(ro - lo, ri - li) <= 8)).to(tl.int32), 0) > 0
    return safe, fit, lo, ro, li, ri


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
):
    mask = (
        (a[:, None] < A) & (b[None, :] < B) & ((left + SITE)[:, None] < right[:, None])
    )
    x = tl.load(X + b[None, :] * N + (left + SITE)[:, None], mask, 0)
    f = tl.load(Co + (DIM * 18 + SITE) * A + slot, a < A, 0)
    v = x * f[:, None]
    if PRIME:
        df = tl.load(Co + (DIM * 18 + 8 + SITE) * A + slot, a < A, 0)
        p = x * df[:, None]
    else:
        p = tl.full(v.shape, 0, tl.float32)
    return v, p


@tr.jit
def _query(
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
):
    if UNROLL:
        v0, p0 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 0, PRIME)
        v1, p1 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 1, PRIME)
        s0, t0 = v0 + v1, p0 + p1
        v2, p2 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 2, PRIME)
        v3, p3 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 3, PRIME)
        s1, t1 = v2 + v3, p2 + p3
        v4, p4 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 4, PRIME)
        v5, p5 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 5, PRIME)
        s2, t2 = v4 + v5, p4 + p5
        v6, p6 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 6, PRIME)
        v7, p7 = _term(X, Co, slot, a, b, left, right, A, B, N, DIM, 7, PRIME)
        value, prime = (s0 + s1) + (s2 + (v6 + v7)), (t0 + t1) + (t2 + (p6 + p7))
    else:
        site = tl.arange(0, 8)
        j = left[:, None] + site[None, :]
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        x = tl.load(X + b[None, :, None] * N + j[:, None, :], mask, 0)
        f = tl.load(
            Co + (DIM * 18 + site[None, :]) * A + slot[:, None], a[:, None] < A, 0
        )
        value = tl.sum(x * f[:, None, :], 2)
        if PRIME:
            df = tl.load(
                Co + (DIM * 18 + 8 + site[None, :]) * A + slot[:, None],
                a[:, None] < A,
                0,
            )
            prime = tl.sum(x * df[:, None, :], 2)
        else:
            prime = tl.full((AB, BB), 0, tl.float32)
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _inject(
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
):
    scale = tl.load(Meta + 12 * a, a < A, 0) * tl.load(Meta + 12 * a + 4, a < A, 0)
    weight = (Value * scale[:, None]).to(tl.float32)
    if UNROLL:
        for site in tl.static_range(8):
            f = tl.load(Co + (DIM * 18 + site) * A + slot, a < A, 0)
            j = left + site
            mask = (a[:, None] < A) & (b[None, :] < B) & (j[:, None] < right[:, None])
            tl.atomic_add(
                Y + b[None, :] * N + j[:, None],
                weight * f[:, None],
                mask,
                sem="relaxed",
            )
    else:
        site = tl.arange(0, 8)
        j = left[:, None] + site[None, :]
        f = tl.load(
            Co + (DIM * 18 + site[None, :]) * A + slot[:, None], a[:, None] < A, 0
        )
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        tl.atomic_add(
            Y + b[None, :, None] * N + j[:, None, :],
            weight[:, :, None] * f[:, None, :],
            mask,
            sem="relaxed",
        )


@tr.jit
def forward(
    X,
    Y,
    Meta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    CORE32: tl.constexpr,
    Order=None,
    CachePos=None,
    CacheCo=None,
    CACHE_UNROLL: tl.constexpr = True,
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    for start in range(base, count, tl.num_programs(0) * AB):
        slot = start + tl.arange(0, AB)
        a = _source(Order, tl.where(slot < count, slot, A), A)
        safe, fit, lo, ro, li, ri = _bounds(CachePos, slot, count, A)
        if CORE32:
            if safe:
                if fit:
                    b = tl.arange(0, BB)
                    h, _ = _query(
                        X,
                        CacheCo,
                        slot,
                        a,
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
                        CACHE_UNROLL,
                    )
                    _inject(
                        h,
                        Y,
                        Meta,
                        CacheCo,
                        slot,
                        a,
                        b,
                        lo,
                        ro,
                        A,
                        B,
                        NO,
                        0,
                        CACHE_UNROLL,
                    )
                else:
                    mk.forward(
                        X,
                        Y,
                        Meta,
                        Count,
                        Origin,
                        Spacing,
                        A,
                        B,
                        NO,
                        NI,
                        AB,
                        BB,
                        BS,
                        RECIPROCAL,
                        True,
                        Base=start,
                        Order=Order,
                    )
        else:
            if not safe:
                mk.forward(
                    X,
                    Y,
                    Meta,
                    Count,
                    Origin,
                    Spacing,
                    A,
                    B,
                    NO,
                    NI,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    False,
                    Base=start,
                    Order=Order,
                )


@tr.jit
def backward(
    X,
    DY,
    DX,
    DP,
    Meta,
    Count,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    NEED_DX: tl.constexpr,
    NEED_DP: tl.constexpr,
    CORE32: tl.constexpr,
    CONTRACT32: tl.constexpr,
    Order=None,
    CachePos=None,
    CacheCo=None,
    CACHE_UNROLL: tl.constexpr = True,
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    for start in range(base, count, tl.num_programs(0) * AB):
        slot = start + tl.arange(0, AB)
        a = _source(Order, tl.where(slot < count, slot, A), A)
        safe, fit, lo, ro, li, ri = _bounds(CachePos, slot, count, A)
        if CORE32:
            if safe:
                if fit:
                    b = tl.arange(0, BB)
                    g, gp = _query(
                        DY,
                        CacheCo,
                        slot,
                        a,
                        b,
                        lo,
                        ro,
                        A,
                        B,
                        NO,
                        0,
                        AB,
                        BB,
                        NEED_DP,
                        CACHE_UNROLL,
                    )
                    if NEED_DP:
                        h, hp = _query(
                            X,
                            CacheCo,
                            slot,
                            a,
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
                            CACHE_UNROLL,
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
                            a,
                            tl.sum(h * g, 1).to(tl.float64),
                            tl.sum(h * gp, 1).to(tl.float64),
                            tl.sum(hp * g, 1).to(tl.float64),
                            A,
                        )
                    if NEED_DX:
                        _inject(
                            g,
                            DX,
                            Meta,
                            CacheCo,
                            slot,
                            a,
                            b,
                            li,
                            ri,
                            A,
                            B,
                            NI,
                            1,
                            CACHE_UNROLL,
                        )
                else:
                    mk.backward(
                        X,
                        DY,
                        DX,
                        DP,
                        Meta,
                        Count,
                        Origin,
                        Spacing,
                        A,
                        B,
                        NO,
                        NI,
                        AB,
                        BB,
                        BS,
                        RECIPROCAL,
                        NEED_DX,
                        NEED_DP,
                        True,
                        CONTRACT32,
                        Base=start,
                        Order=Order,
                    )
        else:
            if not safe:
                mk.backward(
                    X,
                    DY,
                    DX,
                    DP,
                    Meta,
                    Count,
                    Origin,
                    Spacing,
                    A,
                    B,
                    NO,
                    NI,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    NEED_DX,
                    NEED_DP,
                    False,
                    CONTRACT32,
                    Base=start,
                    Order=Order,
                )
