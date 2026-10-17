"""Atom-grouped axis contractions and compact enumeration of support pairs."""

import triton as tr
import triton.language as tl

from .kernels import _coeff, _interval, _shape


@tr.jit
def _atom_geometry(Meta, a, A: tl.constexpr):
    r = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    ho = tl.load(Meta + a * 12 + 10, a < A, 1)
    hi = tl.load(Meta + a * 12 + 11, a < A, 1)
    return r, ir, co, ci, ho, hi


@tr.jit
def _pullback(Meta, DP, a, total, dout, din, A: tl.constexpr):
    inv = tl.load(Meta + a * 12 + 4, a < A, 0)
    scale = tl.load(Meta + a * 12, a < A, 0) * inv
    tl.store(DP + 4 * a, total * inv * tl.load(Meta + a * 12 + 7, a < A, 0), a < A)
    tl.store(DP + 4 * a + 1, total * inv * tl.load(Meta + a * 12 + 8, a < A, 0), a < A)
    tl.store(
        DP + 4 * a + 2,
        scale * (dout - total * tl.load(Meta + a * 12 + 5, a < A, 0)),
        a < A,
    )
    tl.store(
        DP + 4 * a + 3,
        scale * (din - total * tl.load(Meta + a * 12 + 6, a < A, 0)),
        a < A,
    )


@tr.jit
def matrix_flat(
    Meta,
    Matrix,
    DP,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    PAIRS: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    VJP: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    radius, ir, co, ci, ho, hi = _atom_geometry(Meta, a, A)
    lo, _ = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, ho)
    _, ro = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, ho)
    li, _ = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, hi)
    _, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, hi)
    count_i = tl.maximum(ri - li, 1)
    count = tl.where(a < A, (ro - lo) * (ri - li), 0)
    oo, oi = tl.load(Origin).to(tl.float64), tl.load(Origin + 1).to(tl.float64)
    so, si = tl.load(Spacing).to(tl.float64), tl.load(Spacing + 1).to(tl.float64)
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    total = tl.full((AB,), 0, tl.float64)
    dout = tl.full((AB,), 0, tl.float64)
    din = tl.full((AB,), 0, tl.float64)
    for start in range(0, tl.max(count, 0), PAIRS):
        pair = start + tl.arange(0, PAIRS)
        j = lo[:, None] + pair[None, :] // count_i[:, None]
        i = li[:, None] + pair[None, :] % count_i[:, None]
        fo, dfo = _shape(
            oo + j.to(tl.float64) * so - co[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        fi, dfi = _shape(
            oi + i.to(tl.float64) * si - ci[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        mask = (a[:, None] < A) & (pair[None, :] < count[:, None])
        address = Matrix + j * NI + i
        if VJP:
            dw = tl.load(address, mask, 0).to(tl.float64)
            weighted = dw * fi
            total += tl.sum(weighted * fo, 1)
            dout += tl.sum(weighted * dfo, 1)
            din += tl.sum((dw * fo) * dfi, 1)
        else:
            tl.atomic_add(
                address, (scale[:, None] * fo * fi).to(tl.float32), mask, sem="relaxed"
            )
    if VJP:
        _pullback(Meta, DP, a, total, dout, din, A)


@tr.jit
def query_grouped(
    X,
    Bank,
    Meta,
    H,
    HP,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    CUT: tl.constexpr,
    BS: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    o, h = tl.load(Origin + DIM).to(tl.float64), tl.load(Spacing + DIM).to(tl.float64)
    left, _ = _interval(center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, ih)
    _, right = _interval(center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, ih)
    wide = (right - left > CUT) & (a < A)
    small_count = tl.where((a < A) & ~wide, right - left, 0)
    result = tl.full((BB, AB), 0, tl.float64)
    prime = tl.full((BB, AB), 0, tl.float64)
    for start in range(0, tl.max(small_count, 0), BS):
        j = left[:, None] + start + tl.arange(0, BS)[None, :]
        f, dc = _shape(
            o + j.to(tl.float64) * h - center[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        native_mask = (
            (b[:, None, None] < B)
            & (a[None, :, None] < A)
            & ~wide[None, :, None]
            & (j[None, :, :] < right[None, :, None])
        )
        xx = tl.load(X + b[:, None, None] * N + j[None, :, :], native_mask, 0).to(
            tl.float64
        )
        result += tl.sum(xx * f[None, :, :], 2)
        prime += tl.sum(xx * dc[None, :, :], 2)
    if tl.sum(wide.to(tl.int32), 0) > 0:
        half = h * (N - 1) * 0.5
        mid = o + half - center
        for piece in tl.static_range(3):
            l, r = _interval(
                center, radius, Origin, Spacing, N, DIM, piece, RECIPROCAL, ih
            )
            a0, a1, a2, b0, b1 = _coeff(mid, half, radius, ir, piece, RECIPROCAL)
            wide_mask = (b[:, None] < B) & wide[None, :] & (r[None, :] > l[None, :])
            v0 = tl.load(
                Bank + b[:, None] * (N + 1) + r[None, :], wide_mask, 0
            ) - tl.load(Bank + b[:, None] * (N + 1) + l[None, :], wide_mask, 0)
            v1 = tl.load(
                Bank + (B + b[:, None]) * (N + 1) + r[None, :], wide_mask, 0
            ) - tl.load(Bank + (B + b[:, None]) * (N + 1) + l[None, :], wide_mask, 0)
            v2 = tl.load(
                Bank + (2 * B + b[:, None]) * (N + 1) + r[None, :], wide_mask, 0
            ) - tl.load(
                Bank + (2 * B + b[:, None]) * (N + 1) + l[None, :], wide_mask, 0
            )
            result += a0[None, :] * v0 + a1[None, :] * v1 + a2[None, :] * v2
            prime += b0[None, :] * v0 + b1[None, :] * v1
    store_mask = (b[:, None] < B) & (a[None, :] < A)
    tl.store(H + b[:, None] * A + a[None, :], result, store_mask)
    tl.store(HP + b[:, None] * A + a[None, :], prime, store_mask)


@tr.jit
def inject_grouped(
    H,
    Meta,
    Delta,
    Y,
    Origin,
    Spacing,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    DIM: tl.constexpr,
    CUT: tl.constexpr,
    BS: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    ir = tl.load(Meta + a * 12 + 9, a < A, 1)
    ih = tl.load(Meta + a * 12 + 10 + DIM, a < A, 1)
    center = tl.load(Meta + a * 12 + 2 + DIM, a < A, 0)
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    source_mask = (b[:, None] < B) & (a[None, :] < A)
    w = (
        tl.load(H + b[:, None] * A + a[None, :], source_mask, 0).to(tl.float64)
        * scale[None, :]
    )
    o, h = tl.load(Origin + DIM).to(tl.float64), tl.load(Spacing + DIM).to(tl.float64)
    left, _ = _interval(center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, ih)
    _, right = _interval(center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, ih)
    wide = (right - left > CUT) & (a < A)
    small_count = tl.where((a < A) & ~wide, right - left, 0)
    for start in range(0, tl.max(small_count, 0), BS):
        j = left[:, None] + start + tl.arange(0, BS)[None, :]
        f, _native_prime = _shape(
            o + j.to(tl.float64) * h - center[:, None],
            radius[:, None],
            ir[:, None],
            RECIPROCAL,
        )
        native_mask = (
            (b[:, None, None] < B)
            & (a[None, :, None] < A)
            & ~wide[None, :, None]
            & (j[None, :, :] < right[None, :, None])
        )
        tl.atomic_add(
            Y + b[:, None, None] * N + j[None, :, :],
            (w[:, :, None] * f[None, :, :]).to(tl.float32),
            native_mask,
            sem="relaxed",
        )
    if tl.sum(wide.to(tl.int32), 0) > 0:
        half = h * (N - 1) * 0.5
        for piece in tl.static_range(3):
            l, r = _interval(
                center, radius, Origin, Spacing, N, DIM, piece, RECIPROCAL, ih
            )
            a0, a1, a2, _center0, _center1 = _coeff(
                o + half - center, half, radius, ir, piece, RECIPROCAL
            )
            for k in tl.static_range(3):
                coeff = a0 if k == 0 else a1 if k == 1 else a2
                wide_mask = (b[:, None] < B) & wide[None, :] & (r[None, :] > l[None, :])
                tl.atomic_add(
                    Delta + (k * B + b[:, None]) * (N + 1) + l[None, :],
                    w * coeff[None, :],
                    wide_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    Delta + (k * B + b[:, None]) * (N + 1) + r[None, :],
                    -w * coeff[None, :],
                    wide_mask,
                    sem="relaxed",
                )


@tr.jit
def vjp_grouped(
    H,
    HP,
    G,
    GP,
    Meta,
    DP,
    A: tl.constexpr,
    B: tl.constexpr,
    AB: tl.constexpr,
    BB: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    b = tl.arange(0, BB)
    mask = (b[:, None] < B) & (a[None, :] < A)
    offset = b[:, None] * A + a[None, :]
    h = tl.load(H + offset, mask, 0).to(tl.float64)
    hp = tl.load(HP + offset, mask, 0).to(tl.float64)
    g = tl.load(G + offset, mask, 0).to(tl.float64)
    gp = tl.load(GP + offset, mask, 0).to(tl.float64)
    _pullback(Meta, DP, a, tl.sum(h * g, 0), tl.sum(h * gp, 0), tl.sum(hp * g, 0), A)
