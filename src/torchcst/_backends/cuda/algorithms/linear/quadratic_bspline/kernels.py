"""Fused live geometry, exact sampled norms, moment queries and adjoint scatter."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib


@tr.jit
def _coeff(mid, half, radius, piece: tl.constexpr):
    z = mid / radius
    v = half / radius
    if piece == 1:
        a0, a1, a2 = 1.0 - 3.0 * z * z, -6.0 * z * v, -3.0 * v * v
        b0, b1 = 6.0 * z / radius, 6.0 * v / radius
    else:
        sign: tl.constexpr = 1.0 if piece == 0 else -1.0
        base = 1.0 + sign * z
        a0, a1, a2 = 1.5 * base * base, 3.0 * sign * base * v, 1.5 * v * v
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
):
    o = tl.load(Origin + dim).to(tl.float64)
    h = tl.load(Spacing + dim).to(tl.float64)
    low: tl.constexpr = -1.0 if piece == 0 else -1.0 / 3.0 if piece == 1 else 1.0 / 3.0
    high: tl.constexpr = -1.0 / 3.0 if piece == 0 else 1.0 / 3.0 if piece == 1 else 1.0
    left = tl.minimum(tl.maximum(tl.ceil((center + radius * low - o) / h), 0), N).to(
        tl.int32
    )
    right = tl.minimum(tl.maximum(tl.ceil((center + radius * high - o) / h), 0), N).to(
        tl.int32
    )
    return left, right


@tr.jit
def _norm(center, radius, Origin, Spacing, N: tl.constexpr, dim: tl.constexpr):
    q = tl.full((), 0, tl.float64)
    vjp = tl.full((), 0, tl.float64)
    o = tl.load(Origin + dim).to(tl.float64)
    h = tl.load(Spacing + dim).to(tl.float64)
    for piece in tl.static_range(3):
        left, right = _interval(center, radius, Origin, Spacing, N, dim, piece)
        count = (right - left).to(tl.float64)
        safe = tl.maximum(count, 1)
        mid = o + (left.to(tl.float64) + (count - 1) * 0.5) * h - center
        a0, a1, a2, b0, b1 = _coeff(mid, h * count * 0.5, radius, piece)
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
    qo, vo = _norm(co, radius, Origin, Spacing, NO, 0)
    qi, vi = _norm(ci, radius, Origin, Spacing, NI, 1)
    denominator = lib.sqrt(qo * qi)
    correction = tl.where(r2 >= 1.1754943508222875e-38, z0 * inv / safe, 0)
    tl.store(Meta + a * 9, amp, a < A)
    tl.store(Meta + a * 9 + 1, radius, a < A)
    tl.store(Meta + a * 9 + 2, co, a < A)
    tl.store(Meta + a * 9 + 3, ci, a < A)
    tl.store(Meta + a * 9 + 4, 1 / tl.maximum(denominator, FLOOR), a < A)
    tl.store(
        Meta + a * 9 + 5,
        tl.where(denominator >= FLOOR, vo / tl.where(qo > 0, qo, 1), 0),
        a < A,
    )
    tl.store(
        Meta + a * 9 + 6,
        tl.where(denominator >= FLOOR, vi / tl.where(qi > 0, qi, 1), 0),
        a < A,
    )
    tl.store(Meta + a * 9 + 7, maximum_amp * (inv - z0 * correction), a < A)
    tl.store(Meta + a * 9 + 8, -maximum_amp * z1 * correction, a < A)


@tr.jit
def _shape(d, radius):
    u = d / radius
    gap = tl.maximum(1 - tl.abs(u), 0)
    outer = 1.5 * gap * gap
    inner = 1 - 3 * u * u
    f = tl.where(tl.abs(u) <= 1.0 / 3.0, inner, outer)
    dc = tl.where(
        tl.abs(u) <= 1.0 / 3.0,
        6 * u / radius,
        3 * tl.where(u >= 0, 1.0, -1.0) * tl.maximum(1 - tl.abs(u), 0) / radius,
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
):
    a = tl.program_id(0)
    b = tl.program_id(1) * 16 + tl.arange(0, 16)
    radius = tl.load(Meta + a * 9 + 1)
    center = tl.load(Meta + a * 9 + 2 + DIM)
    o = tl.load(Origin + DIM).to(tl.float64)
    h = tl.load(Spacing + DIM).to(tl.float64)
    left, _first_right = _interval(center, radius, Origin, Spacing, N, DIM, 0)
    _last_left, right = _interval(center, radius, Origin, Spacing, N, DIM, 2)
    result = tl.full((16,), 0, tl.float64)
    prime = tl.full((16,), 0, tl.float64)
    if DIRECT or right - left <= CUT:
        for start in range(left, right, 64):
            j = start + tl.arange(0, 64)
            f, dc = _shape(o + j.to(tl.float64) * h - center, radius)
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
            l, r = _interval(center, radius, Origin, Spacing, N, DIM, piece)
            a0, a1, a2, b0, b1 = _coeff(mid, half, radius, piece)
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
):
    a = tl.program_id(0)
    b = tl.program_id(1) * 16 + tl.arange(0, 16)
    radius = tl.load(Meta + a * 9 + 1)
    center = tl.load(Meta + a * 9 + 2 + DIM)
    scale = tl.load(Meta + a * 9) * tl.load(Meta + a * 9 + 4)
    w = tl.load(H + b * A + a, b < B, 0).to(tl.float64) * scale
    left, _first_right = _interval(center, radius, Origin, Spacing, N, DIM, 0)
    _last_left, right = _interval(center, radius, Origin, Spacing, N, DIM, 2)
    o = tl.load(Origin + DIM).to(tl.float64)
    h = tl.load(Spacing + DIM).to(tl.float64)
    if DIRECT or right - left <= CUT:
        for start in range(left, right, 64):
            j = start + tl.arange(0, 64)
            f, _dc = _shape(o + j.to(tl.float64) * h - center, radius)
            tl.atomic_add(
                Y + b[:, None] * N + j[None, :],
                (w[:, None] * f[None, :]).to(tl.float32),
                (b[:, None] < B) & (j[None, :] < right),
                sem="relaxed",
            )
    else:
        half = h * (N - 1) * 0.5
        for piece in tl.static_range(3):
            l, r = _interval(center, radius, Origin, Spacing, N, DIM, piece)
            a0, a1, a2, _b0, _b1 = _coeff(o + half - center, half, radius, piece)
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
    amp = tl.load(Meta + a * 9)
    inv = tl.load(Meta + a * 9 + 4)
    tl.store(DP + 4 * a, hg * inv * tl.load(Meta + a * 9 + 7))
    tl.store(DP + 4 * a + 1, hg * inv * tl.load(Meta + a * 9 + 8))
    tl.store(
        DP + 4 * a + 2, amp * inv * (tl.sum(h * gp, 0) - hg * tl.load(Meta + a * 9 + 5))
    )
    tl.store(
        DP + 4 * a + 3, amp * inv * (tl.sum(hp * g, 0) - hg * tl.load(Meta + a * 9 + 6))
    )
