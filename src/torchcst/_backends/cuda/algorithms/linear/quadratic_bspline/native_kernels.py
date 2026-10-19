"""Small-support axis H/G contractions without W, dW or GEMM workspace."""

import triton as tr
import triton.language as tl

from .kernels import _interval, _shape
from .parallel_kernels import _pullback


@tr.jit
def _axis(
    Meta,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    origin = tl.load(Origin + DIM).to(tl.float64)
    spacing = tl.load(Spacing + DIM).to(tl.float64)
    left, _ = _interval(center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, ih)
    _, right = _interval(center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, ih)
    left, right = tl.where(a < A, left, 0), tl.where(a < A, right, 0)
    return radius, ir, center, origin, spacing, left, right


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
    radius, ir, center, origin, spacing, left, right = _axis(
        Meta, Origin, Spacing, a, A, N, DIM, RECIPROCAL
    )
    sites = tl.arange(0, BS)
    value = tl.full((AB, BB), 0, tl.float64)
    prime = tl.full((AB, BB), 0, tl.float64)
    for start in range(0, tl.max(right - left, 0), BS):
        j = left[:, None] + start + sites[None, :]
        f, df = _shape(
            origin + j.to(tl.float64) * spacing - center[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        x = tl.load(X + b[None, :, None] * N + j[:, None, :], mask, 0).to(tl.float64)
        value += tl.sum(x * f[:, None, :], 2)
        if PRIME:
            prime += tl.sum(x * df[:, None, :], 2)
    # Same stored-H precision boundary as the broad endpoint route.
    return value.to(tl.float32).to(tl.float64), prime.to(tl.float32).to(tl.float64)


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
    radius, ir, center, origin, spacing, left, right = _axis(
        Meta, Origin, Spacing, a, A, N, DIM, RECIPROCAL
    )
    sites = tl.arange(0, BS)
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    weighted = Value * scale[:, None]
    for start in range(0, tl.max(right - left, 0), BS):
        j = left[:, None] + start + sites[None, :]
        f, _ = _shape(
            origin + j.to(tl.float64) * spacing - center[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        mask = (
            (a[:, None, None] < A)
            & (b[None, :, None] < B)
            & (j[:, None, :] < right[:, None, None])
        )
        v = weighted[:, :, None] * f[:, None, :]
        tl.atomic_add(
            Y + b[None, :, None] * N + j[:, None, :],
            v.to(Y.dtype.element_ty),
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
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    if base < count:
        slot = base + tl.arange(0, AB)
        a = tl.where(slot < count, slot, A)
        b = tl.arange(0, BB)
        h, _ = _query(
            X, Meta, Origin, Spacing, a, b, A, B, NI, 1, AB, BB, BS, RECIPROCAL, False
        )
        _inject(h, Y, Meta, Origin, Spacing, a, b, A, B, NO, 0, BS, RECIPROCAL)


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
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    if base < count:
        slot = base + tl.arange(0, AB)
        a = tl.where(slot < count, slot, A)
        b = tl.arange(0, BB)
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
        if NEED_DP:
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
            _pullback(
                Meta, DP, a, tl.sum(h * g, 1), tl.sum(h * gp, 1), tl.sum(hp * g, 1), A
            )
        if NEED_DX:
            _inject(g, DX, Meta, Origin, Spacing, a, b, A, B, NI, 1, BS, RECIPROCAL)


@tr.jit
def _cached_axis(
    Meta,
    Origin,
    Spacing,
    a,
    A: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    radius, ir, center, origin, spacing, left, right = _axis(
        Meta, Origin, Spacing, a, A, N, DIM, RECIPROCAL
    )
    j = left[:, None] + tl.arange(0, BS)[None, :]
    f, df = _shape(
        origin + j.to(tl.float64) * spacing - center[:, None],
        radius[:, None],
        ir[:, None],
        RECIPROCAL,
    )
    valid = (a[:, None] < A) & (j < right[:, None])
    return (
        j,
        tl.where(valid, f, 0),
        tl.where(valid, df, 0),
        right,
        tl.max(right - left, 0) <= BS,
    )


@tr.jit
def _cached_query(
    X,
    j,
    f,
    df,
    right,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    PRIME: tl.constexpr,
):
    mask = (
        (a[:, None, None] < A)
        & (b[None, :, None] < B)
        & (j[:, None, :] < right[:, None, None])
    )
    xx = tl.load(X + b[None, :, None] * N + j[:, None, :], mask, 0).to(tl.float64)
    h = tl.sum(xx * f[:, None, :], 2).to(tl.float32).to(tl.float64)
    hp = tl.full(h.shape, 0, tl.float64)
    if PRIME:
        hp = tl.sum(xx * df[:, None, :], 2).to(tl.float32).to(tl.float64)
    return h, hp


@tr.jit
def _cached_inject(
    Value,
    Y,
    j,
    f,
    right,
    scale,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
):
    mask = (
        (a[:, None, None] < A)
        & (b[None, :, None] < B)
        & (j[:, None, :] < right[:, None, None])
    )
    value = (Value * scale[:, None])[:, :, None] * f[:, None, :]
    tl.atomic_add(
        Y + b[None, :, None] * N + j[:, None, :],
        value.to(tl.float32),
        mask,
        sem="relaxed",
    )


@tr.jit
def forward_loop(
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
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    if base < count:
        a = tl.where(base + tl.arange(0, AB) < count, base + tl.arange(0, AB), A)
        ji, fi, dfi, ri, cached_i = _cached_axis(
            Meta, Origin, Spacing, a, A, NI, 1, BS, RECIPROCAL
        )
        jo, fo, _dfo, ro, cached_o = _cached_axis(
            Meta, Origin, Spacing, a, A, NO, 0, BS, RECIPROCAL
        )
        scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
        for part in range(tl.cdiv(B, BB)):
            b = part * BB + tl.arange(0, BB)
            if cached_i:
                h, _ = _cached_query(X, ji, fi, dfi, ri, a, b, A, B, NI, False)
            else:
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
            if cached_o:
                _cached_inject(h, Y, jo, fo, ro, scale, a, b, A, B, NO)
            else:
                _inject(h, Y, Meta, Origin, Spacing, a, b, A, B, NO, 0, BS, RECIPROCAL)


@tr.jit
def backward_loop(
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
):
    base = tl.program_id(0) * AB
    count = tl.load(Count)
    if base < count:
        a = tl.where(base + tl.arange(0, AB) < count, base + tl.arange(0, AB), A)
        ji, fi, dfi, ri, cached_i = _cached_axis(
            Meta, Origin, Spacing, a, A, NI, 1, BS, RECIPROCAL
        )
        jo, fo, dfo, ro, cached_o = _cached_axis(
            Meta, Origin, Spacing, a, A, NO, 0, BS, RECIPROCAL
        )
        scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
        total = tl.full((AB,), 0, tl.float64)
        dout = tl.full((AB,), 0, tl.float64)
        din = tl.full((AB,), 0, tl.float64)
        for part in range(tl.cdiv(B, BB)):
            b = part * BB + tl.arange(0, BB)
            if cached_o:
                g, gp = _cached_query(DY, jo, fo, dfo, ro, a, b, A, B, NO, NEED_DP)
            else:
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
            if NEED_DP:
                if cached_i:
                    h, hp = _cached_query(X, ji, fi, dfi, ri, a, b, A, B, NI, True)
                else:
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
                total += tl.sum(h * g, 1)
                dout += tl.sum(h * gp, 1)
                din += tl.sum(hp * g, 1)
            if NEED_DX:
                if cached_i:
                    _cached_inject(g, DX, ji, fi, ri, scale, a, b, A, B, NI)
                else:
                    _inject(
                        g, DX, Meta, Origin, Spacing, a, b, A, B, NI, 1, BS, RECIPROCAL
                    )
        if NEED_DP:
            _pullback(Meta, DP, a, total, dout, din, A)
