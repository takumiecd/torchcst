"""Cyclic two-state exponential scans, adjacent-width queries and adjoints."""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib


@tr.jit
def compose(e1, a1, v01, v11, e2, a2, v02, v12):
    return e1 * e2, a1 + a2, e2 * v01 + v02, e2 * (v11 + a2 * v01) + v12


@tr.jit
def local_scan(
    X,
    C,
    Z,
    Summary,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
    H: tl.constexpr,
    INJECT: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
):
    tile = tl.program_id(0)
    bb = tl.program_id(1)
    jd = tl.program_id(2)
    j = jd % J
    direction = jd // J
    b = bb * BB + tl.arange(0, BB)
    l = tile * BS + tl.arange(0, BS)
    site = tl.where(direction == 0, l, N - 1 - l)
    lam = tl.load(C + j * 7)
    a = (lam * H).to(tl.float32)
    e = tl.load(C + j * 7 + 4).to(tl.float32)
    if INJECT:
        base = (((1 - direction) * J + j) * N + site[None, :]) * B + b[:, None]
        v0 = tl.load(X + base * 2 + 1, b[:, None] < B, 0)
        v1 = tl.load(X + base * 2, b[:, None] < B, 0)
    else:
        v0 = tl.load(X + b[:, None] * SX0 + site[None, :] * SX1, b[:, None] < B, 0)
        v1 = tl.full((BB, BS), 0, tl.float32)
    es = tl.full((BB, BS), 1, tl.float32) * e
    as_ = tl.full((BB, BS), 1, tl.float32) * a
    _, _, v0, v1 = tl.associative_scan((es, as_, v0, v1), 1, compose)
    base = ((jd * N + site[None, :]) * B + b[:, None]) * 2
    tl.store(Z + base, v0, b[:, None] < B)
    tl.store(Z + base + 1, v1, b[:, None] < B)
    last = tl.arange(0, BS) == BS - 1
    end0 = tl.sum(tl.where(last[None, :], v0, 0), 1)
    end1 = tl.sum(tl.where(last[None, :], v1, 0), 1)
    sb = ((jd * (N // BS) + tile) * B + b) * 2
    tl.store(Summary + sb, end0, b < B)
    tl.store(Summary + sb + 1, end1, b < B)


@tr.jit
def stitch(
    C,
    Summary,
    Start,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
    TT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    bb = tl.program_id(0)
    jd = tl.program_id(1)
    j = jd % J
    b = bb * BB + tl.arange(0, BB)
    t = tl.arange(0, TT)
    nt = N // BS
    lam = tl.load(C + j * 7)
    a = (lam * BS * H).to(tl.float32)
    e = lib.exp(-lam * BS * H).to(tl.float32)
    base = ((jd * nt + t[None, :]) * B + b[:, None]) * 2
    v0 = tl.load(Summary + base, (b[:, None] < B) & (t[None, :] < nt), 0)
    v1 = tl.load(Summary + base + 1, (b[:, None] < B) & (t[None, :] < nt), 0)
    es = tl.broadcast_to(tl.where(t[None, :] < nt, e, 1), (BB, TT))
    aa = tl.broadcast_to(tl.where(t[None, :] < nt, a, 0), (BB, TT))
    _, _, v0, v1 = tl.associative_scan((es, aa, v0, v1), 1, compose)
    total0 = tl.sum(tl.where(t[None, :] == nt - 1, v0, 0), 1)
    total1 = tl.sum(tl.where(t[None, :] == nt - 1, v1, 0), 1)
    q = tl.load(C + j * 7 + 5).to(tl.float32)
    one = tl.load(C + j * 7 + 6).to(tl.float32)
    z0 = total0 / one
    z1 = total1 / one + (lam * D).to(tl.float32) * q * total0 / (one * one)
    prev = tl.broadcast_to(tl.maximum(t[None, :] - 1, 0), (BB, TT))
    prev0 = tl.gather(v0, prev, 1)
    prev1 = tl.gather(v1, prev, 1)
    prev0 = tl.where(t[None, :] == 0, 0, prev0)
    prev1 = tl.where(t[None, :] == 0, 0, prev1)
    decay = tl.exp(-a * t.to(tl.float32))
    dist = a * t.to(tl.float32)
    incoming0 = prev0 + decay[None, :] * z0[:, None]
    incoming1 = prev1 + decay[None, :] * (z1[:, None] + dist[None, :] * z0[:, None])
    tl.store(Start + base, incoming0, (b[:, None] < B) & (t[None, :] < nt))
    tl.store(Start + base + 1, incoming1, (b[:, None] < B) & (t[None, :] < nt))


@tr.jit
def finish_scan(
    C,
    Z,
    Start,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
    H: tl.constexpr,
):
    tile = tl.program_id(0)
    bb = tl.program_id(1)
    jd = tl.program_id(2)
    j = jd % J
    direction = jd // J
    b = bb * BB + tl.arange(0, BB)
    l = tile * BS + tl.arange(0, BS)
    site = tl.where(direction == 0, l, N - 1 - l)
    base = ((jd * N + site[None, :]) * B + b[:, None]) * 2
    v0 = tl.load(Z + base, b[:, None] < B, 0)
    v1 = tl.load(Z + base + 1, b[:, None] < B, 0)
    start = ((jd * (N // BS) + tile) * B + b) * 2
    z0 = tl.load(Start + start, b < B, 0)
    z1 = tl.load(Start + start + 1, b < B, 0)
    a = (tl.load(C + j * 7) * H).to(tl.float32)
    dist = (tl.arange(0, BS) + 1).to(tl.float32) * a
    decay = tl.exp(-dist)
    tl.store(Z + base, v0 + decay[None, :] * z0[:, None], b[:, None] < B)
    tl.store(
        Z + base + 1,
        v1 + decay[None, :] * (z1[:, None] + dist[None, :] * z0[:, None]),
        b[:, None] < B,
    )


@tr.jit
def gram(C, M, row, j, k, delta, N: tl.constexpr, D: tl.constexpr, H: tl.constexpr):
    lj = tl.load(C + j * 7)
    lk = tl.load(C + k * 7)
    aj = tl.load(C + j * 7 + 1)
    ak = tl.load(C + k * 7 + 1)
    bj = tl.load(C + j * 7 + 2)
    bk = tl.load(C + k * 7 + 2)
    pj = tl.load(C + j * 7 + 3)
    pk = tl.load(C + k * 7 + 3)
    value = tl.full(delta.shape, 0, tl.float64)
    prime = tl.full(delta.shape, 0, tl.float64)
    for o in tl.static_range(4):
        sj = 1 if o < 2 else -1
        sk = 1 if o % 2 == 0 else -1
        rate = sj * lj + sk * lk
        reverse = rate < 0
        u0 = tl.where(
            reverse, tl.full((), (N - 1) * H, tl.float64), tl.full((), 0, tl.float64)
        )
        direction = tl.where(reverse, -1, 1)
        rj = delta + u0 if sj == 1 else D - delta - u0
        rk = delta + u0 if sk == 1 else D - delta - u0
        xj = aj + bj * rj
        xk = ak + bk * rk
        yj = direction * sj * bj
        yk = direction * sk * bk
        s0 = tl.load(M + row * 12 + o * 3)
        s1 = tl.load(M + row * 12 + o * 3 + 1)
        s2 = tl.load(M + row * 12 + o * 3 + 2)
        polynomial = xj * xk * s0 + (xj * yk + xk * yj) * s1 + yj * yk * s2
        pp = (sj * bj * xk + sk * bk * xj) * s0 + (sj * bj * yk + sk * bk * yj) * s1
        weight = lib.exp(-(lj * rj + lk * rk)) / (pj * pk)
        value += weight * polynomial
        prime += weight * (pp - rate * polynomial)
    return value, prime


@tr.jit
def prepare_geometry(
    P,
    Jidx,
    T,
    C,
    M,
    Index,
    Delta,
    Norm,
    Prime,
    A: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    Origin,
    DIM: tl.constexpr,
    BA: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    valid = a < A
    p = tl.load(P + a * 4 + 2 + DIM, valid, 0).to(tl.float64) - tl.load(Origin + DIM)
    p = p - lib.floor(p / D) * D
    right = lib.ceil(p / H).to(tl.int32)
    delta = tl.minimum(tl.maximum(right.to(tl.float64) * H - p, 0), H)
    j = tl.load(Jidx + a, valid, 0).to(tl.int32)
    t = tl.load(T + a, valid, 0)
    g00, p00 = gram(C, M, 2 * j, j, j, delta, N, D, H)
    g01, p01 = gram(C, M, 2 * j + 1, j, j + 1, delta, N, D, H)
    g11, p11 = gram(C, M, 2 * j + 2, j + 1, j + 1, delta, N, D, H)
    norm = (1 - t) * (1 - t) * g00 + 2 * t * (1 - t) * g01 + t * t * g11
    prime = -((1 - t) * (1 - t) * p00 + 2 * t * (1 - t) * p01 + t * t * p11)
    tl.store(Index + a * 3, j, valid)
    tl.store(Index + a * 3 + 1, (right + N - 1) % N, valid)
    tl.store(Index + a * 3 + 2, right % N, valid)
    tl.store(Delta + a, delta, valid)
    tl.store(Norm + a, norm, valid)
    tl.store(Prime + a, prime, valid)


@tr.jit
def prepare_coeff(
    Index,
    Delta,
    T,
    C,
    Norm,
    Prime,
    Other,
    Coeff,
    A: tl.constexpr,
    H: tl.constexpr,
    FLOOR: tl.constexpr,
    BA: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    valid = a < A
    j = tl.load(Index + a * 3, valid, 0)
    delta = tl.load(Delta + a, valid, 0)
    t = tl.load(T + a, valid, 0)
    n = tl.load(Norm + a, valid, 1)
    np_ = tl.load(Prime + a, valid, 0)
    other = tl.load(Other + a, valid, 1)
    root = lib.sqrt(n)
    corr = tl.where(lib.sqrt(n * other) >= FLOOR, np_ / (2 * n), 0)
    for w in tl.static_range(2):
        lam = tl.load(C + (j + w) * 7)
        peak = tl.load(C + (j + w) * 7 + 3)
        mix = 1 - t if w == 0 else t
        for direction in tl.static_range(2):
            dist = H - delta if direction == 0 else delta
            z = dist * lam
            factor = lib.exp(-z) * mix / (peak * root)
            c0 = factor * (1 + z)
            c1 = factor
            sign = -1 if direction == 0 else 1
            p0 = sign * lam * factor * z - corr * c0
            p1 = sign * lam * factor - corr * c1
            slot = w * 4 + direction * 2
            tl.store(Coeff + a * 16 + slot, c0.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + slot + 1, c1.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + 8 + slot, p0.to(tl.float32), valid)
            tl.store(Coeff + a * 16 + 8 + slot + 1, p1.to(tl.float32), valid)


@tr.jit
def query(
    Z,
    Index,
    Coeff,
    Y,
    Prime,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    j = tl.load(Index + a * 3, a < A, 0)
    out = tl.full((BB, BA), 0, tl.float32)
    prime = tl.full((BB, BA), 0, tl.float32)
    for w in tl.static_range(2):
        for direction in tl.static_range(2):
            site = tl.load(Index + a * 3 + 1 + direction, a < A, 0)
            base = (
                ((direction * J + j[None, :] + w) * N + site[None, :]) * B + b[:, None]
            ) * 2
            z0 = tl.load(Z + base, (a[None, :] < A) & (b[:, None] < B), 0)
            z1 = tl.load(Z + base + 1, (a[None, :] < A) & (b[:, None] < B), 0)
            slot = w * 4 + direction * 2
            c0 = tl.load(Coeff + a * 16 + slot, a < A, 0)
            c1 = tl.load(Coeff + a * 16 + slot + 1, a < A, 0)
            p0 = tl.load(Coeff + a * 16 + slot + 8, a < A, 0)
            p1 = tl.load(Coeff + a * 16 + slot + 9, a < A, 0)
            out += z0 * c0[None, :] + z1 * c1[None, :]
            prime += z0 * p0[None, :] + z1 * p1[None, :]
    offset = b[:, None] * A + a[None, :]
    mask = (a[None, :] < A) & (b[:, None] < B)
    tl.store(Y + offset, out, mask)
    tl.store(Prime + offset, prime, mask)


@tr.jit
def inject(
    V,
    Amp,
    Index,
    Coeff,
    Out,
    A: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    v = (
        tl.load(V + b[:, None] * A + a[None, :], (b[:, None] < B) & (a[None, :] < A), 0)
        * tl.load(Amp + a, a < A, 0)[None, :]
    )
    j = tl.load(Index + a * 3, a < A, 0)
    for w in tl.static_range(2):
        for direction in tl.static_range(2):
            site = tl.load(Index + a * 3 + 1 + direction, a < A, 0)
            base = (
                ((direction * J + j[None, :] + w) * N + site[None, :]) * B + b[:, None]
            ) * 2
            slot = w * 4 + direction * 2
            c0 = tl.load(Coeff + a * 16 + slot, a < A, 0)
            c1 = tl.load(Coeff + a * 16 + slot + 1, a < A, 0)
            mask = (a[None, :] < A) & (b[:, None] < B)
            tl.atomic_add(Out + base, v * c0[None, :], mask, sem="relaxed")
            tl.atomic_add(Out + base + 1, v * c1[None, :], mask, sem="relaxed")


@tr.jit
def restore_sum(
    Z,
    Y,
    N: tl.constexpr,
    B: tl.constexpr,
    J: tl.constexpr,
    BS: tl.constexpr,
    BB: tl.constexpr,
):
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    out = tl.full((BB, BS), 0, tl.float32)
    for jd in tl.static_range(2 * J):
        base = ((jd * N + s[None, :]) * B + b[:, None]) * 2
        out += tl.load(Z + base + 1, (s[None, :] < N) & (b[:, None] < B), 0)
    tl.store(Y + b[:, None] * N + s[None, :], out, (s[None, :] < N) & (b[:, None] < B))


@tr.jit
def direct_axis(
    P,
    Index,
    T,
    C,
    Norm,
    Prime,
    Other,
    F,
    Fp,
    A: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    Origin,
    DIM: tl.constexpr,
    FLOOR: tl.constexpr,
    BS: tl.constexpr,
):
    a = tl.program_id(0)
    s = tl.program_id(1) * BS + tl.arange(0, BS)
    centre = tl.load(P + a * 4 + DIM + 2).to(tl.float64) - tl.load(Origin + DIM)
    d = s.to(tl.float64) * H - centre
    d = d - lib.floor(d / D + 0.5) * D
    r = tl.abs(d)
    sign = tl.where(d > 0, 1, tl.where(d < 0, -1, 0))
    j = tl.load(Index + a * 3)
    t = tl.load(T + a)
    norm = tl.load(Norm + a)
    prime = tl.load(Prime + a)
    other = tl.load(Other + a)
    corr = tl.where(lib.sqrt(norm * other) >= FLOOR, prime / (2 * norm), 0)
    out = tl.full((BS,), 0, tl.float64)
    deriv = tl.full((BS,), 0, tl.float64)
    for w in tl.static_range(2):
        lam = tl.load(C + (j + w) * 7)
        aa = tl.load(C + (j + w) * 7 + 1)
        bb = tl.load(C + (j + w) * 7 + 2)
        peak = tl.load(C + (j + w) * 7 + 3)
        e0 = lib.exp(-lam * r)
        e1 = lib.exp(-lam * (D - r))
        v = (e0 * (aa + bb * r) + e1 * (aa + bb * (D - r))) / peak
        vp = (
            -sign
            * (e0 * (bb - lam * (aa + bb * r)) - e1 * (bb - lam * (aa + bb * (D - r))))
            / peak
        )
        mix = 1 - t if w == 0 else t
        out += mix * v
        deriv += mix * vp
    out = out / lib.sqrt(norm)
    deriv = deriv / lib.sqrt(norm) - corr * out
    tl.store(F + a * N + s, out.to(tl.float32), s < N)
    tl.store(Fp + a * N + s, deriv.to(tl.float32), s < N)


@tr.jit
def parameter_vjp(
    H, Hp, G, Gp, Amp, Grad, A: tl.constexpr, B: tl.constexpr, BB: tl.constexpr
):
    a = tl.program_id(0)
    b = tl.arange(0, BB)
    h = tl.load(H + b * A + a, b < B, 0)
    hp = tl.load(Hp + b * A + a, b < B, 0)
    g = tl.load(G + b * A + a, b < B, 0)
    gp = tl.load(Gp + b * A + a, b < B, 0)
    amp = tl.load(Amp + a)
    tl.store(Grad + a * 3, tl.sum(h * g, 0))
    tl.store(Grad + a * 3 + 1, amp * tl.sum(h * gp, 0))
    tl.store(Grad + a * 3 + 2, amp * tl.sum(hp * g, 0))
