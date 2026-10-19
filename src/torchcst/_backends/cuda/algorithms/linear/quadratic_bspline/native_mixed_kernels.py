"""Local-coordinate FP32 native core, with a separately launched FP64 guard."""

import triton as tr
import triton.language as tl

from . import native_kernels as nk
from .kernels import _shape
from .parallel_kernels import _pullback


@tr.jit
def _safe(
    Meta,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    _, _, _, _, _, lo, ro = nk._axis(Meta, Origin, Spacing, a, A, NO, 0, RECIPROCAL)
    _, _, _, _, _, li, ri = nk._axis(Meta, Origin, Spacing, a, A, NI, 1, RECIPROCAL)
    ok = (a >= A) | (tl.minimum(ro - lo, ri - li) > 2)
    return tl.min(ok.to(tl.int32), 0) > 0


@tr.jit
def _query(
    X,
    Meta,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    PRIME: tl.constexpr,
):
    radius, ir, center, origin, spacing, left, right = nk._axis(
        Meta, Origin, Spacing, a, A, N, DIM, RECIPROCAL
    )
    distance = (origin + left.to(tl.float64) * spacing - center).to(tl.float32)
    spacing32 = spacing.to(tl.float32)
    radius32, ir32 = radius.to(tl.float32), ir.to(tl.float32)
    sites = tl.arange(0, BS)
    value = tl.full((AB, BB), 0, tl.float32)
    prime = tl.full((AB, BB), 0, tl.float32)
    for start in range(0, tl.max(right - left, 0), BS):
        j = left[:, None] + start + sites[None, :]
        d = distance[:, None] + (start + sites[None, :]).to(tl.float32) * spacing32
        f, df = _shape(d, radius32[:, None], ir32[:, None], RECIPROCAL)
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        xx = tl.load(X + b[None, :, None] * N + j[:, None, :], mask, 0)
        value += tl.sum(xx * f[:, None, :], 2)
        if PRIME:
            prime += tl.sum(xx * df[:, None, :], 2)
    return value.to(tl.float64), prime.to(tl.float64)


@tr.jit
def _inject(
    Value,
    Y,
    Meta,
    Origin,
    Spacing,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    radius, ir, center, origin, spacing, left, right = nk._axis(
        Meta, Origin, Spacing, a, A, N, DIM, RECIPROCAL
    )
    distance = (origin + left.to(tl.float64) * spacing - center).to(tl.float32)
    spacing32 = spacing.to(tl.float32)
    radius32, ir32 = radius.to(tl.float32), ir.to(tl.float32)
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    weighted = (Value * scale[:, None]).to(tl.float32)
    sites = tl.arange(0, BS)
    for start in range(0, tl.max(right - left, 0), BS):
        j = left[:, None] + start + sites[None, :]
        d = distance[:, None] + (start + sites[None, :]).to(tl.float32) * spacing32
        f, _ = _shape(d, radius32[:, None], ir32[:, None], RECIPROCAL)
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        tl.atomic_add(
            Y + b[None, :, None] * N + j[:, None, :],
            weighted[:, :, None] * f[:, None, :],
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
    Base=None,
    Order=None,
):
    base = tl.program_id(0) * AB if Base is None else Base
    count = tl.load(Count)
    if base < count:
        slot = base + tl.arange(0, AB)
        a = tl.where(slot < count, slot, A)
        if Order is not None:
            a = tl.load(Order + slot, slot < count, A).to(tl.int32)
        b = tl.arange(0, BB)
        safe = _safe(Meta, Origin, Spacing, a, A, NO, NI, RECIPROCAL)
        active = safe if CORE32 else ~safe
        if active:
            if CORE32:
                h, _ = _query(
                    X,
                    Meta,
                    Origin,
                    Spacing,
                    a,
                    b,
                    A,
                    B,
                    NI,
                    1,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    False,
                )
                _inject(h, Y, Meta, Origin, Spacing, a, b, A, B, NO, 0, BS, RECIPROCAL)
            else:
                h, _ = nk._query(
                    X,
                    Meta,
                    Origin,
                    Spacing,
                    a,
                    b,
                    A,
                    B,
                    NI,
                    1,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    False,
                    KEEP64=True,
                )
                nk._inject(
                    h, Y, Meta, Origin, Spacing, a, b, A, B, NO, 0, BS, RECIPROCAL
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
    Base=None,
    Order=None,
):
    base = tl.program_id(0) * AB if Base is None else Base
    count = tl.load(Count)
    if base < count:
        slot = base + tl.arange(0, AB)
        a = tl.where(slot < count, slot, A)
        if Order is not None:
            a = tl.load(Order + slot, slot < count, A).to(tl.int32)
        b = tl.arange(0, BB)
        safe = _safe(Meta, Origin, Spacing, a, A, NO, NI, RECIPROCAL)
        active = safe if CORE32 else ~safe
        if active:
            if CORE32:
                g, gp = _query(
                    DY,
                    Meta,
                    Origin,
                    Spacing,
                    a,
                    b,
                    A,
                    B,
                    NO,
                    0,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    NEED_DP,
                )
            else:
                g, gp = nk._query(
                    DY,
                    Meta,
                    Origin,
                    Spacing,
                    a,
                    b,
                    A,
                    B,
                    NO,
                    0,
                    AB,
                    BB,
                    BS,
                    RECIPROCAL,
                    NEED_DP,
                    KEEP64=True,
                )
            if NEED_DP:
                if CORE32:
                    h, hp = _query(
                        X,
                        Meta,
                        Origin,
                        Spacing,
                        a,
                        b,
                        A,
                        B,
                        NI,
                        1,
                        AB,
                        BB,
                        BS,
                        RECIPROCAL,
                        True,
                    )
                else:
                    h, hp = nk._query(
                        X,
                        Meta,
                        Origin,
                        Spacing,
                        a,
                        b,
                        A,
                        B,
                        NI,
                        1,
                        AB,
                        BB,
                        BS,
                        RECIPROCAL,
                        True,
                        KEEP64=True,
                    )
                if CORE32 and CONTRACT32:
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
                if CORE32:
                    _inject(
                        g, DX, Meta, Origin, Spacing, a, b, A, B, NI, 1, BS, RECIPROCAL
                    )
                else:
                    nk._inject(
                        g, DX, Meta, Origin, Spacing, a, b, A, B, NI, 1, BS, RECIPROCAL
                    )
