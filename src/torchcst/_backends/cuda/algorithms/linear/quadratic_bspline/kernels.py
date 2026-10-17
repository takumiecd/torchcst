"""Fused live geometry, exact sampled norms, moment queries and adjoint scatter."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _coeff(
    mid, half, radius, inverse_radius, piece: tl.constexpr, RECIPROCAL: tl.constexpr
):
    if RECIPROCAL:
        z = mid * inverse_radius
        v = half * inverse_radius
    else:
        z = mid / radius
        v = half / radius
    if piece == 1:
        a0, a1, a2 = 1.0 - 3.0 * z * z, -6.0 * z * v, -3.0 * v * v
        if RECIPROCAL:
            b0, b1 = 6.0 * z * inverse_radius, 6.0 * v * inverse_radius
        else:
            b0, b1 = 6.0 * z / radius, 6.0 * v / radius
    else:
        sign: tl.constexpr = 1.0 if piece == 0 else -1.0
        base = 1.0 + sign * z
        a0, a1, a2 = 1.5 * base * base, 3.0 * sign * base * v, 1.5 * v * v
        if RECIPROCAL:
            b0, b1 = -3.0 * sign * base * inverse_radius, -3.0 * v * inverse_radius
        else:
            b0, b1 = -3.0 * sign * base / radius, -3.0 * v / radius
    return a0, a1, a2, b0, b1


@tr.jit
def _interval(
    center,
    radius,
    Origin,
    Spacing,
    N: tl.constexpr,
    dim: tl.constexpr,
    piece: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    inverse_h,
):
    o = tl.load(Origin + dim).to(tl.float64)
    h = tl.load(Spacing + dim).to(tl.float64)
    low: tl.constexpr = -1.0 if piece == 0 else -1.0 / 3.0 if piece == 1 else 1.0 / 3.0
    high: tl.constexpr = -1.0 / 3.0 if piece == 0 else 1.0 / 3.0 if piece == 1 else 1.0
    if RECIPROCAL:
        low_index = (center + radius * low - o) * inverse_h
        high_index = (center + radius * high - o) * inverse_h
    else:
        low_index = (center + radius * low - o) / h
        high_index = (center + radius * high - o) / h
    left = tl.minimum(tl.maximum(tl.ceil(low_index), 0), N).to(tl.int32)
    right = tl.minimum(tl.maximum(tl.ceil(high_index), 0), N).to(tl.int32)
    return left, right


@tr.jit
def _norm(
    center,
    radius,
    Origin,
    Spacing,
    Moments,
    N: tl.constexpr,
    dim: tl.constexpr,
    ML: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    LOOKUP: tl.constexpr,
):
    q = tl.full((), 0, tl.float64)
    vjp = tl.full((), 0, tl.float64)
    o = tl.load(Origin + dim).to(tl.float64)
    h = tl.load(Spacing + dim).to(tl.float64)
    inverse_radius = 1.0 / radius
    inverse_h = 1.0 / h
    for piece in tl.static_range(3):
        left, right = _interval(
            center, radius, Origin, Spacing, N, dim, piece, RECIPROCAL, inverse_h
        )
        count = (right - left).to(tl.float64)
        safe = tl.maximum(count, 1)
        mid = o + (left.to(tl.float64) + (count - 1) * 0.5) * h - center
        a0, a1, a2, b0, b1 = _coeff(
            mid, h * count * 0.5, radius, inverse_radius, piece, RECIPROCAL
        )
        if LOOKUP:
            index = (right - left).to(tl.int32)
            m2 = tl.load(Moments + index)
            m4 = tl.load(Moments + ML + index)
        else:
            m2 = tl.where(count > 1, count / 3 - 1 / (3 * safe), 0)
            m4 = tl.where(
                count > 1, count / 5 - 2 / (3 * safe) + 7 / (15 * safe * safe * safe), 0
            )
        q += a0 * a0 * count + (a1 * a1 + 2 * a0 * a2) * m2 + a2 * a2 * m4
        vjp += a0 * b0 * count + (a1 * b1 + a2 * b0) * m2
    return q, vjp


@tr.jit
def prepare(
    P,
    Meta,
    Origin,
    Spacing,
    Scalars,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    FLOOR: tl.constexpr,
    Moments,
    ML: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    LOOKUP: tl.constexpr,
):
    a = tl.program_id(0) * 64 + tl.arange(0, 64)
    z0 = tl.load(P + 4 * a, a < A, 0).to(tl.float64)
    z1 = tl.load(P + 4 * a + 1, a < A, 0).to(tl.float64)
    r2 = z0 * z0 + z1 * z1
    safe = tl.maximum(r2, 1.1754943508222875e-38)
    inv = 1 / lib.sqrt(safe)
    maximum_amp = tl.load(Scalars[0]).to(tl.float64)
    amp = maximum_amp * z0 * inv
    alpha = tl.minimum(tl.maximum((r2 - 1) / 3, 0), 1)
    w = amp / tl.load(Scalars[1]).to(tl.float64)
    x = w * w
    minimum = tl.load(Scalars[5]).to(tl.float64)
    birth = tl.load(Scalars[6]).to(tl.float64)
    maximum = tl.load(Scalars[7]).to(tl.float64)
    kappa = tl.load(Scalars[2]).to(tl.float64)
    power = tl.load(Scalars[4]).to(tl.float64)
    if power == 1.0:
        ux = x
    else:
        ux = lib.pow(x, power)
    upper = minimum + (maximum - minimum) * kappa / (kappa + ux)
    upper = tl.maximum(upper, tl.load(Scalars[8]).to(tl.float64))
    lower = minimum + (birth - minimum) / (1 + tl.load(Scalars[3]).to(tl.float64) * x)
    upper = tl.maximum(upper, lower)
    sigma = lib.exp((1 - alpha) * lib.log(lower) + alpha * lib.log(upper))
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    radius = 2.449489742783178 * sigma
    co = tl.load(P + 4 * a + 2, a < A, 0).to(tl.float64)
    ci = tl.load(P + 4 * a + 3, a < A, 0).to(tl.float64)
    qo, vo = _norm(co, radius, Origin, Spacing, Moments, NO, 0, ML, RECIPROCAL, LOOKUP)
    qi, vi = _norm(ci, radius, Origin, Spacing, Moments, NI, 1, ML, RECIPROCAL, LOOKUP)
    denominator = lib.sqrt(qo * qi)
    correction = tl.where(r2 >= 1.1754943508222875e-38, z0 * inv / safe, 0)
    tl.store(Meta + a * 12, amp, a < A)
    tl.store(Meta + a * 12 + 1, radius, a < A)
    tl.store(Meta + a * 12 + 2, co, a < A)
    tl.store(Meta + a * 12 + 3, ci, a < A)
    tl.store(Meta + a * 12 + 4, 1 / tl.maximum(denominator, FLOOR), a < A)
    tl.store(
        Meta + a * 12 + 5,
        tl.where(denominator >= FLOOR, vo / tl.where(qo > 0, qo, 1), 0),
        a < A,
    )
    tl.store(
        Meta + a * 12 + 6,
        tl.where(denominator >= FLOOR, vi / tl.where(qi > 0, qi, 1), 0),
        a < A,
    )
    tl.store(Meta + a * 12 + 7, maximum_amp * (inv - z0 * correction), a < A)
    tl.store(Meta + a * 12 + 8, -maximum_amp * z1 * correction, a < A)
    tl.store(Meta + a * 12 + 9, 1.0 / radius, a < A)
    tl.store(Meta + a * 12 + 10, 1.0 / tl.load(Spacing).to(tl.float64), a < A)
    tl.store(Meta + a * 12 + 11, 1.0 / tl.load(Spacing + 1).to(tl.float64), a < A)


@tr.jit
def norm_replay(
    Meta,
    Origin,
    Spacing,
    Moments,
    Scratch,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    ML: tl.constexpr,
    FLOOR: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    LOOKUP: tl.constexpr,
):
    # Diagnostic re-evaluation from already decoded live geometry. Not on the
    # normal execution path; its timing is not subtracted from fused prepare.
    a = tl.program_id(0) * 64 + tl.arange(0, 64)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    qo, vo = _norm(co, radius, Origin, Spacing, Moments, NO, 0, ML, RECIPROCAL, LOOKUP)
    qi, vi = _norm(ci, radius, Origin, Spacing, Moments, NI, 1, ML, RECIPROCAL, LOOKUP)
    d = lib.sqrt(qo * qi)
    tl.store(Scratch + 4 * a, 1 / tl.maximum(d, FLOOR), a < A)
    tl.store(
        Scratch + 4 * a + 1,
        tl.where(d >= FLOOR, vo / tl.where(qo > 0, qo, 1), 0),
        a < A,
    )
    tl.store(
        Scratch + 4 * a + 2,
        tl.where(d >= FLOOR, vi / tl.where(qi > 0, qi, 1), 0),
        a < A,
    )
    tl.store(Scratch + 4 * a + 3, qo + qi, a < A)


@tr.jit
def _shape(d, radius, inverse_radius, RECIPROCAL: tl.constexpr):
    if RECIPROCAL:
        inverse = inverse_radius
        u = d * inverse
    else:
        u = d / radius
    gap = tl.maximum(1 - tl.abs(u), 0)
    outer = 1.5 * gap * gap
    inner = 1 - 3 * u * u
    f = tl.where(tl.abs(u) <= 1.0 / 3.0, inner, outer)
    if RECIPROCAL:
        dc = tl.where(
            tl.abs(u) <= 1.0 / 3.0,
            6 * u * inverse,
            3 * tl.where(u >= 0, 1.0, -1.0) * gap * inverse,
        )
    else:
        dc = tl.where(
            tl.abs(u) <= 1.0 / 3.0,
            6 * u / radius,
            3 * tl.where(u >= 0, 1.0, -1.0) * gap / radius,
        )
    return f, dc


@tr.jit
def query(
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
    DIRECT: tl.constexpr,
    CUT: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1) * 16 + tl.arange(0, 16)
    radius = tl.load(Meta + a * 12 + 1)
    inverse_radius = tl.load(Meta + a * 12 + 9)
    inverse_h = tl.load(Meta + a * 12 + 10 + DIM)
    center = tl.load(Meta + a * 12 + 2 + DIM)
    o = tl.load(Origin + DIM).to(tl.float64)
    h = tl.load(Spacing + DIM).to(tl.float64)
    left, _first_right = _interval(
        center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, inverse_h
    )
    _last_left, right = _interval(
        center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, inverse_h
    )
    result = tl.full((16,), 0, tl.float64)
    prime = tl.full((16,), 0, tl.float64)
    if DIRECT or right - left <= CUT:
        for start in range(left, right, BS):
            j = start + tl.arange(0, BS)
            f, dc = _shape(
                o + j.to(tl.float64) * h - center, radius, inverse_radius, RECIPROCAL
            )
            xx = tl.load(
                X + b[:, None] * N + j[None, :],
                (b[:, None] < B) & (j[None, :] < right),
                0,
            ).to(tl.float64)
            result += tl.sum(xx * f[None, :], 1)
            prime += tl.sum(xx * dc[None, :], 1)
    else:
        half = h * (N - 1) * 0.5
        mid = o + half - center
        for piece in tl.static_range(3):
            l, r = _interval(
                center, radius, Origin, Spacing, N, DIM, piece, RECIPROCAL, inverse_h
            )
            a0, a1, a2, b0, b1 = _coeff(
                mid, half, radius, inverse_radius, piece, RECIPROCAL
            )
            v0 = tl.load(Bank + b * (N + 1) + r, b < B, 0) - tl.load(
                Bank + b * (N + 1) + l, b < B, 0
            )
            v1 = tl.load(Bank + (B + b) * (N + 1) + r, b < B, 0) - tl.load(
                Bank + (B + b) * (N + 1) + l, b < B, 0
            )
            v2 = tl.load(Bank + (2 * B + b) * (N + 1) + r, b < B, 0) - tl.load(
                Bank + (2 * B + b) * (N + 1) + l, b < B, 0
            )
            result += a0 * v0 + a1 * v1 + a2 * v2
            prime += b0 * v0 + b1 * v1
    tl.store(H + b * A + a, result, b < B)
    tl.store(HP + b * A + a, prime, b < B)


@tr.jit
def inject(
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
    DIRECT: tl.constexpr,
    CUT: tl.constexpr,
    BS: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1) * 16 + tl.arange(0, 16)
    radius = tl.load(Meta + a * 12 + 1)
    inverse_radius = tl.load(Meta + a * 12 + 9)
    inverse_h = tl.load(Meta + a * 12 + 10 + DIM)
    center = tl.load(Meta + a * 12 + 2 + DIM)
    scale = tl.load(Meta + a * 12) * tl.load(Meta + a * 12 + 4)
    w = tl.load(H + b * A + a, b < B, 0).to(tl.float64) * scale
    left, _first_right = _interval(
        center, radius, Origin, Spacing, N, DIM, 0, RECIPROCAL, inverse_h
    )
    _last_left, right = _interval(
        center, radius, Origin, Spacing, N, DIM, 2, RECIPROCAL, inverse_h
    )
    o = tl.load(Origin + DIM).to(tl.float64)
    h = tl.load(Spacing + DIM).to(tl.float64)
    if DIRECT or right - left <= CUT:
        for start in range(left, right, BS):
            j = start + tl.arange(0, BS)
            f, _dc = _shape(
                o + j.to(tl.float64) * h - center, radius, inverse_radius, RECIPROCAL
            )
            tl.atomic_add(
                Y + b[:, None] * N + j[None, :],
                (w[:, None] * f[None, :]).to(tl.float32),
                (b[:, None] < B) & (j[None, :] < right),
                sem="relaxed",
            )
    else:
        half = h * (N - 1) * 0.5
        for piece in tl.static_range(3):
            l, r = _interval(
                center, radius, Origin, Spacing, N, DIM, piece, RECIPROCAL, inverse_h
            )
            a0, a1, a2, _b0, _b1 = _coeff(
                o + half - center, half, radius, inverse_radius, piece, RECIPROCAL
            )
            for k in tl.static_range(3):
                coeff = a0 if k == 0 else a1 if k == 1 else a2
                tl.atomic_add(
                    Delta + (k * B + b) * (N + 1) + l, w * coeff, b < B, sem="relaxed"
                )
                tl.atomic_add(
                    Delta + (k * B + b) * (N + 1) + r, -w * coeff, b < B, sem="relaxed"
                )


@tr.jit
def restore(Scan, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * 128 + tl.arange(0, 128)
    j = idx % N
    b = idx // N
    z = j.to(tl.float64) * (2.0 / (N - 1)) - 1
    v0 = tl.load(Scan + b * (N + 1) + j, b < B, 0)
    v1 = tl.load(Scan + (B + b) * (N + 1) + j, b < B, 0)
    v2 = tl.load(Scan + (2 * B + b) * (N + 1) + j, b < B, 0)
    old = tl.load(Y + idx, b < B, 0)
    tl.store(Y + idx, old + v0 + z * v1 + z * z * v2, b < B)


@tr.jit
def vjp(H, HP, G, GP, Meta, DP, A: tl.constexpr, B: tl.constexpr, BB: tl.constexpr):
    a = tl.program_id(0)
    b = tl.arange(0, BB)
    h = tl.load(H + b * A + a, b < B, 0).to(tl.float64)
    hp = tl.load(HP + b * A + a, b < B, 0).to(tl.float64)
    g = tl.load(G + b * A + a, b < B, 0).to(tl.float64)
    gp = tl.load(GP + b * A + a, b < B, 0).to(tl.float64)
    hg = tl.sum(h * g, 0)
    amp = tl.load(Meta + a * 12)
    inv = tl.load(Meta + a * 12 + 4)
    tl.store(DP + 4 * a, hg * inv * tl.load(Meta + a * 12 + 7))
    tl.store(DP + 4 * a + 1, hg * inv * tl.load(Meta + a * 12 + 8))
    tl.store(
        DP + 4 * a + 2,
        amp * inv * (tl.sum(h * gp, 0) - hg * tl.load(Meta + a * 12 + 5)),
    )
    tl.store(
        DP + 4 * a + 3,
        amp * inv * (tl.sum(hp * g, 0) - hg * tl.load(Meta + a * 12 + 6)),
    )


@tr.jit
def matrix_build(
    Meta,
    W,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    BS: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    inverse_radius = tl.load(Meta + a * 12 + 9, a < A, 1)
    inverse_ho = tl.load(Meta + a * 12 + 10, a < A, 1)
    inverse_hi = tl.load(Meta + a * 12 + 11, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    scale = tl.load(Meta + a * 12, a < A, 0) * tl.load(Meta + a * 12 + 4, a < A, 0)
    lo, _ro = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, inverse_ho)
    _lo, hi = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, inverse_ho)
    li, _ri = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, inverse_hi)
    _li, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, inverse_hi)
    rows = tl.arange(0, BS)
    oo = tl.load(Origin).to(tl.float64)
    oi = tl.load(Origin + 1).to(tl.float64)
    so = tl.load(Spacing).to(tl.float64)
    si = tl.load(Spacing + 1).to(tl.float64)
    for start_o in range(0, tl.max(hi - lo, 0), BS):
        j = lo[:, None] + start_o + rows[None, :]
        fo, _dfo = _shape(
            oo + j.to(tl.float64) * so - co[:, None],
            radius[:, None],
            inverse_radius[:, None],
            RECIPROCAL,
        )
        for start_i in range(0, tl.max(ri - li, 0), BS):
            i = li[:, None] + start_i + rows[None, :]
            fi, _dfi = _shape(
                oi + i.to(tl.float64) * si - ci[:, None],
                radius[:, None],
                inverse_radius[:, None],
                RECIPROCAL,
            )
            w = scale[:, None, None] * fo[:, :, None] * fi[:, None, :]
            mask = (
                (a[:, None, None] < A)
                & (j[:, :, None] < hi[:, None, None])
                & (i[:, None, :] < ri[:, None, None])
            )
            tl.atomic_add(
                W + j[:, :, None] * NI + i[:, None, :],
                w.to(tl.float32),
                mask,
                sem="relaxed",
            )


@tr.jit
def matrix_vjp(
    Meta,
    DW,
    DP,
    Origin,
    Spacing,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    BS: tl.constexpr,
    AB: tl.constexpr,
    RECIPROCAL: tl.constexpr,
):
    a = tl.program_id(0) * AB + tl.arange(0, AB)
    radius = tl.load(Meta + a * 12 + 1, a < A, 1)
    inverse_radius = tl.load(Meta + a * 12 + 9, a < A, 1)
    inverse_ho = tl.load(Meta + a * 12 + 10, a < A, 1)
    inverse_hi = tl.load(Meta + a * 12 + 11, a < A, 1)
    co = tl.load(Meta + a * 12 + 2, a < A, 0)
    ci = tl.load(Meta + a * 12 + 3, a < A, 0)
    lo, _ro = _interval(co, radius, Origin, Spacing, NO, 0, 0, RECIPROCAL, inverse_ho)
    _lo, hi = _interval(co, radius, Origin, Spacing, NO, 0, 2, RECIPROCAL, inverse_ho)
    li, _ri = _interval(ci, radius, Origin, Spacing, NI, 1, 0, RECIPROCAL, inverse_hi)
    _li, ri = _interval(ci, radius, Origin, Spacing, NI, 1, 2, RECIPROCAL, inverse_hi)
    rows = tl.arange(0, BS)
    oo = tl.load(Origin).to(tl.float64)
    oi = tl.load(Origin + 1).to(tl.float64)
    so = tl.load(Spacing).to(tl.float64)
    si = tl.load(Spacing + 1).to(tl.float64)
    total = tl.full((AB,), 0, tl.float64)
    dout = tl.full((AB,), 0, tl.float64)
    din = tl.full((AB,), 0, tl.float64)
    for start_o in range(0, tl.max(hi - lo, 0), BS):
        j = lo[:, None] + start_o + rows[None, :]
        fo, dfo = _shape(
            oo + j.to(tl.float64) * so - co[:, None],
            radius[:, None],
            inverse_radius[:, None],
            RECIPROCAL,
        )
        for start_i in range(0, tl.max(ri - li, 0), BS):
            i = li[:, None] + start_i + rows[None, :]
            fi, dfi = _shape(
                oi + i.to(tl.float64) * si - ci[:, None],
                radius[:, None],
                inverse_radius[:, None],
                RECIPROCAL,
            )
            mask = (
                (a[:, None, None] < A)
                & (j[:, :, None] < hi[:, None, None])
                & (i[:, None, :] < ri[:, None, None])
            )
            dw = tl.load(DW + j[:, :, None] * NI + i[:, None, :], mask, 0).to(
                tl.float64
            )
            total += tl.sum(tl.sum(dw * fo[:, :, None] * fi[:, None, :], 2), 1)
            dout += tl.sum(tl.sum(dw * dfo[:, :, None] * fi[:, None, :], 2), 1)
            din += tl.sum(tl.sum(dw * fo[:, :, None] * dfi[:, None, :], 2), 1)
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
