"""Centered-support kernels. FP64 geometry/shape/norm, FP32 shared reductions.

Only complete interior tiles enter Z/C. Every partial endpoint interval is
evaluated from the SAME finite series. No whole-tile support substitution.
"""

import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib


@tr.jit
def _smooth(z):
    z = tl.minimum(tl.maximum(z, 0.0), 1.0)
    return z * z * z * (10 + z * (-15 + 6 * z)), 30 * z * z * (1 - z) * (1 - z)


@tr.jit
def prepare(
    P,
    Matrix,
    Offset,
    Lookup,
    Meta,
    Co,
    Ci,
    A: tl.constexpr,
    M: tl.constexpr,
    F: tl.constexpr,
    T0: tl.constexpr,
    L0: tl.constexpr,
    A0: tl.constexpr,
    KMAX: tl.constexpr,
    CUTOFF: tl.constexpr,
    SHARED: tl.constexpr,
):
    a = tl.program_id(0)
    logit = tl.load(P + a * 4 + 3).to(tl.float64)
    sig = 1 / (1 + lib.exp(-logit))
    width = 0.1 + 9.9 * sig
    level = tl.full((), 0, tl.int32)
    for j in tl.static_range(1, 12):
        level = tl.where(width >= A0 * (2.0**j), j, level)
    c = tl.program_id(1)
    lev = tl.where(c < 2, tl.maximum(level - 1, 0), level)
    scale = lib.exp2(lev.to(tl.float64))
    threshold = A0 * lib.exp2(level.to(tl.float64))
    interval = tl.where(level > 0, 0.15 * threshold, 1.0)
    h, dh = _smooth((width - threshold) / interval)
    h = tl.where(level > 0, h, 1.0)
    dh = tl.where(level > 0, dh / interval, 0.0)
    outer = tl.where(c < 2, 1 - h, h)
    do = tl.where(c < 2, -dh, dh)
    tile, step = T0 * scale, L0 * scale
    n = lib.ceil((width + tile / 2) / step)
    low = n * step
    w, dw = _smooth((width - low + tile / 2 + 0.25 * step) / (0.25 * step))
    dw = dw / (0.25 * step)
    inner = tl.where(c % 2 == 0, 1 - w, w)
    di = tl.where(c % 2 == 0, -dw, dw)
    weight, wi = outer * inner, do * inner + outer * di
    length = low + (c % 2) * step
    k = ((n + c % 2) * scale).to(tl.int32)
    bank = tl.full((), -1, tl.int32)
    if SHARED:
        bank = tl.load(
            Lookup + lev * (KMAX + 1) + k,
            (k <= KMAX) & (weight > 0) & (width >= CUTOFF),
            -1,
        )
    tl.store(Meta + (a * 4 + c) * 4 + 0, length)
    tl.store(Meta + (a * 4 + c) * 4 + 1, tile)
    tl.store(Meta + (a * 4 + c) * 4 + 2, bank.to(tl.float64))
    tl.store(Meta + (a * 4 + c) * 4 + 3, weight)
    nmode = tl.arange(0, F)
    outb = tl.full((F,), 0.0, tl.float64)
    outi = tl.full((F,), 0.0, tl.float64)
    if (weight > 0) | (wi != 0):
        q = width / length
        z = 3.141592653589793 * q * nmode
        small = tl.abs(z) < 4
        t = tl.where(small, z, 0.0)
        term = tl.full((F,), 256 / 315, tl.float64)
        integ, prime = term, tl.full((F,), 0, tl.float64)
        for m in tl.static_range(20):
            term *= -t * t / ((2 * m + 2) * (2 * m + 11))
            integ += term
            prime += (2 * m + 2) * term
        v = tl.where(small, 4.0, z)
        v2 = v * v
        ss, cc = lib.sin(v), lib.cos(v)
        large = (
            768
            * ((105 - 45 * v2 + v2 * v2) * ss - (105 * v - 10 * v * v2) * cc)
            / (v * v * v * v * v * v * v * v * v)
        )
        largep = (
            -768
            * (
                (945 - 420 * v2 + 15 * v2 * v2) * ss
                - (945 * v - 105 * v * v2 + v * v2 * v2) * cc
            )
            / (v * v * v * v * v * v * v * v * v)
        )
        integ, prime = tl.where(small, integ, large), tl.where(small, prime, largep)
        factor = tl.where(nmode == 0, 0.5, 1.0)
        raw = q * integ * factor
        rawq = (integ + prime) * factor
        mat = tl.load(
            Matrix + nmode[:, None] * (M + 1) + nmode[None, :],
            (nmode[:, None] <= M) & (nmode[None, :] <= M),
            0,
        )
        off = tl.load(Offset + nmode, nmode <= M, 0)
        b = tl.sum(raw[None, :] * mat, 1) + off
        bq = tl.sum(rawq[None, :] * mat, 1)
        outb = b * weight
        outi = bq / length * weight + b * wi
    addr = (a * 4 + c) * (M + 1) + nmode
    tl.store(Co + addr, outb, nmode <= M)
    tl.store(Ci + addr, outi, nmode <= M)


@tr.jit
def _component(s, mu, a, c, Meta, Co, Ci, M: tl.constexpr, GRAD: tl.constexpr):
    length = tl.load(Meta + (a * 4 + c) * 4)
    r = s.to(tl.float64) * 0.1 - mu
    theta = r * (3.141592653589793 / length)
    cs, ss = lib.cos(theta), lib.sin(theta)
    cn = tl.full(s.shape, 1.0, tl.float64)
    sn = tl.full(s.shape, 0.0, tl.float64)
    f = tl.full(s.shape, 0.0, tl.float64)
    fm = tl.full(s.shape, 0.0, tl.float64)
    fi = tl.full(s.shape, 0.0, tl.float64)
    for n in tl.static_range(M + 1):
        b = tl.load(Co + (a * 4 + c) * (M + 1) + n)
        f += b * cn
        if GRAD:
            bi = tl.load(Ci + (a * 4 + c) * (M + 1) + n)
            fm += b * sn * (3.141592653589793 * n / length)
            fi += bi * cn
        next_c = cn * cs - sn * ss
        sn = sn * cs + cn * ss
        cn = next_c
    mask = tl.abs(r) < length
    return tl.where(mask, f, 0.0), tl.where(mask, fm, 0.0), tl.where(mask, fi, 0.0)


@tr.jit
def _bounds(mu, length, N: tl.constexpr):
    lo = tl.minimum(tl.maximum(lib.floor((mu - length) / 0.1).to(tl.int32) + 1, 0), N)
    hi = tl.minimum(tl.maximum(lib.ceil((mu + length) / 0.1).to(tl.int32), 0), N)
    return lo, hi


@tr.jit
def norms(
    P,
    Meta,
    Co,
    Ci,
    Norm,
    A: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    Native=None,
    Bounds=None,
    NC: tl.constexpr = 0,
):
    a, axis = tl.program_id(0), tl.program_id(1)
    mu = tl.load(P + a * 4 + axis + 1).to(tl.float64)
    maxL = tl.full((), 0.0, tl.float64)
    for c in tl.static_range(4):
        L = tl.load(Meta + (a * 4 + c) * 4)
        w = tl.load(Meta + (a * 4 + c) * 4 + 3)
        maxL = tl.maximum(maxL, tl.where(w > 0, L, 0.0))
    lo, hi = _bounds(mu, maxL, N)
    if NC:
        direct = tl.full((), True, tl.int1)
        for c in tl.static_range(4):
            direct = direct & (tl.load(Meta + (a * 4 + c) * 4 + 2) < 0)
        cached = direct & (hi - lo <= NC)
        tl.store(Bounds + (a * 2 + axis) * 2, lo)
        tl.store(Bounds + (a * 2 + axis) * 2 + 1, tl.where(cached, hi - lo, -1))
    acc = tl.full((BS,), 0.0, tl.float64)
    am = tl.full((BS,), 0.0, tl.float64)
    ai = tl.full((BS,), 0.0, tl.float64)
    for start in range(lo, hi, BS):
        s = start + tl.arange(0, BS)
        f = tl.full((BS,), 0.0, tl.float64)
        fm = tl.full((BS,), 0.0, tl.float64)
        fi = tl.full((BS,), 0.0, tl.float64)
        for c in tl.static_range(4):
            if tl.load(Meta + (a * 4 + c) * 4 + 3) > 0:
                v, vm, vi = _component(s, mu, a, c, Meta, Co, Ci, M, True)
                f += v
                fm += vm
                fi += vi
        acc += tl.where(s < hi, f * f, 0.0)
        am += tl.where(s < hi, f * fm, 0.0)
        ai += tl.where(s < hi, f * fi, 0.0)
        if NC:  # noqa: SIM102 - compile-time guard for the optional cache pointers
            if cached:
                index = s - lo
                valid = (s < hi) & (index < NC)
                tl.store(Native + ((a * 2 + axis) * 3) * NC + index, f, valid)
                tl.store(Native + ((a * 2 + axis) * 3 + 1) * NC + index, fm, valid)
                tl.store(Native + ((a * 2 + axis) * 3 + 2) * NC + index, fi, valid)
    tl.store(Norm + (a * 2 + axis) * 3 + 0, tl.sum(acc, 0))
    tl.store(Norm + (a * 2 + axis) * 3 + 1, tl.sum(am, 0))
    tl.store(Norm + (a * 2 + axis) * 3 + 2, tl.sum(ai, 0))


@tr.jit
def pack(
    P,
    Meta,
    Co,
    Ci,
    Beta,
    A: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    F: tl.constexpr,
):
    a, axis = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, 4)
    f = tl.arange(0, F)
    n = tl.where(f <= M, f, f - M)
    L = tl.load(Meta + (a * 4 + c) * 4)
    mu = tl.load(P + a * 4 + axis + 1).to(tl.float64)
    angle = 3.141592653589793 * mu * n[None, :] / L[:, None]
    trig = tl.where(f[None, :] <= M, lib.cos(angle), lib.sin(angle))
    b = tl.load(Co + (a * 4 + c[:, None]) * (M + 1) + n[None, :], f[None, :] < R, 0)
    bi = tl.load(Ci + (a * 4 + c[:, None]) * (M + 1) + n[None, :], f[None, :] < R, 0)
    tl.store(
        Beta + (((axis * A + a) * 4 + c[:, None]) * 2 + 0) * R + f[None, :],
        b * trig,
        f[None, :] < R,
    )
    tl.store(
        Beta + (((axis * A + a) * 4 + c[:, None]) * 2 + 1) * R + f[None, :],
        bi * trig,
        f[None, :] < R,
    )


@tr.jit
def moments(
    X,
    Phi,
    Nodes,
    Z,
    B: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
):
    node = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.arange(0, FF)
    bank = tl.load(Nodes + node * 3)
    start = tl.load(Nodes + node * 3 + 1)
    S = tl.load(Nodes + node * 3 + 2)
    acc = tl.full((BB, FF), 0.0, tl.float32)
    for base in range(0, S, BS):
        s = base + tl.arange(0, BS)
        x = tl.load(
            X + b[:, None] * N + start + s[None, :],
            (b[:, None] < B) & (s[None, :] < S),
            0,
        )
        phi = tl.load(
            Phi + (bank * N + start + s[:, None]) * R + f[None, :],
            (s[:, None] < S) & (f[None, :] < R),
            0,
        )
        acc += tl.dot(x, phi, input_precision="ieee")
    tl.store(
        Z + (b[:, None] * J + node) * R + f[None, :],
        acc,
        (b[:, None] < B) & (f[None, :] < R),
    )


@tr.jit
def contract(
    X,
    P,
    Meta,
    Co,
    Ci,
    Beta,
    Banks,
    Z,
    H,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    AXIS: tl.constexpr,
    GRAD: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    mu = tl.load(P + a * 4 + AXIS + 1).to(tl.float64)
    hv = tl.full((), 0.0, tl.float64)
    hm = tl.full((), 0.0, tl.float64)
    hi_ = tl.full((), 0.0, tl.float64)
    for c in tl.static_range(4):
        if tl.load(Meta + (a * 4 + c) * 4 + 3) > 0:
            L = tl.load(Meta + (a * 4 + c) * 4)
            bank = tl.load(Meta + (a * 4 + c) * 4 + 2).to(tl.int32)
            lo, hi = _bounds(mu, L, N)
            left, right = lo, lo
            if bank >= 0:
                S = tl.load(Banks + bank * 2)
                offset = tl.load(Banks + bank * 2 + 1)
                left = (lo + S - 1) // S
                right = hi // S
                if right > left:
                    f = tl.arange(0, FF)
                    beta = tl.load(
                        Beta + (((AXIS * A + a) * 4 + c) * 2) * R + f, f < R, 0
                    ).to(tl.float64)
                    if GRAD:
                        bi = tl.load(
                            Beta + (((AXIS * A + a) * 4 + c) * 2 + 1) * R + f, f < R, 0
                        ).to(tl.float64)
                        n = tl.where(f <= M, f, f - M)
                        pair = tl.where(f <= M, M + n, n)
                        pb = tl.load(
                            Beta + (((AXIS * A + a) * 4 + c) * 2) * R + pair,
                            (f < R) & (n > 0),
                            0,
                        ).to(tl.float64)
                        bm = tl.where(f <= M, -pb, pb) * (3.141592653589793 * n / L)
                    for tile in range(left, right):
                        z = tl.load(
                            Z + (b * J + offset + tile) * R + f,
                            (b < B) & (f < R),
                            0,
                        ).to(tl.float64)
                        hv += tl.sum(z * beta, 0)
                        if GRAD:
                            hm += tl.sum(z * bm, 0)
                            hi_ += tl.sum(z * bi, 0)
                    left, right = left * S, right * S
                else:
                    left, right = lo, lo
            for part in tl.static_range(2):
                begin = tl.where(part == 0, lo, right)
                end = tl.where(part == 0, left, hi)
                for start in range(begin, end, BS):
                    s = start + tl.arange(0, BS)
                    fv, fm, fi = _component(s, mu, a, c, Meta, Co, Ci, M, GRAD)
                    xv = tl.load(
                        X + b * N + s,
                        (b < B) & (s < end),
                        0,
                    ).to(tl.float64)
                    hv += tl.sum(xv * fv, 0)
                    if GRAD:
                        hm += tl.sum(xv * fm, 0)
                        hi_ += tl.sum(xv * fi, 0)
    tl.store(H + (b * A + a) * 3, hv, b < B)
    if GRAD:
        tl.store(H + (b * A + a) * 3 + 1, hm, b < B)
        tl.store(H + (b * A + a) * 3 + 2, hi_, b < B)


@tr.jit
def _alpha(P, Norm, a):
    no = tl.load(Norm + (a * 2) * 3)
    ni = tl.load(Norm + (a * 2 + 1) * 3)
    denom = tl.maximum(lib.sqrt(no * ni), 1.0e-6)
    return tl.load(P + a * 4).to(tl.float64) / denom


@tr.jit
def contract_cached(
    X,
    P,
    Meta,
    Co,
    Ci,
    Beta,
    Banks,
    Z,
    H,
    Native,
    Bounds,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    AXIS: tl.constexpr,
    GRAD: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
    NC: tl.constexpr,
):
    a, b = tl.program_id(0), tl.program_id(1)
    count = tl.load(Bounds + (a * 2 + AXIS) * 2 + 1)
    if count >= 0:
        lo = tl.load(Bounds + (a * 2 + AXIS) * 2)
        s = tl.arange(0, NC)
        x = tl.load(X + b * N + lo + s, s < count, 0).to(tl.float64)
        f = tl.load(Native + ((a * 2 + AXIS) * 3) * NC + s, s < count, 0).to(tl.float64)
        tl.store(H + (b * A + a) * 3, tl.sum(x * f, 0))
        if GRAD:
            fm = tl.load(Native + ((a * 2 + AXIS) * 3 + 1) * NC + s, s < count, 0).to(
                tl.float64
            )
            fi = tl.load(Native + ((a * 2 + AXIS) * 3 + 2) * NC + s, s < count, 0).to(
                tl.float64
            )
            tl.store(H + (b * A + a) * 3 + 1, tl.sum(x * fm, 0))
            tl.store(H + (b * A + a) * 3 + 2, tl.sum(x * fi, 0))
    else:
        contract(
            X,
            P,
            Meta,
            Co,
            Ci,
            Beta,
            Banks,
            Z,
            H,
            B,
            N,
            A,
            M,
            R,
            J,
            AXIS,
            GRAD,
            BB,
            BS,
            FF,
        )


@tr.jit
def aggregate(
    H,
    P,
    Meta,
    Norm,
    Beta,
    Order,
    Begin,
    Nodes,
    C,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    AXIS: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    FF: tl.constexpr,
):
    node = tl.program_id(0)
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    f = tl.arange(0, FF)
    bank = tl.load(Nodes + node * 3)
    first = tl.load(Nodes + node * 3 + 1)
    S = tl.load(Nodes + node * 3 + 2)
    begin, end = tl.load(Begin + bank), tl.load(Begin + bank + 1)
    acc = tl.full((BB, FF), 0.0, tl.float32)
    for base in range(begin, end, BA):
        pos = base + tl.arange(0, BA)
        idx = tl.load(Order + pos, pos < end, 0)
        a, c = idx // 4, idx % 4
        mu = tl.load(P + a * 4 + AXIS + 1).to(tl.float64)
        L = tl.load(Meta + (a * 4 + c) * 4)
        lo, hi = _bounds(mu, L, N)
        covered = (first >= lo) & (first + S <= hi) & (pos < end)
        alpha = _alpha(P, Norm, a).to(tl.float32)
        h = (
            tl.load(
                H + (b[:, None] * A + a[None, :]) * 3,
                (b[:, None] < B) & covered[None, :],
                0,
            )
            * alpha[None, :]
        )
        beta = tl.load(
            Beta + (((AXIS * A + a[:, None]) * 4 + c[:, None]) * 2) * R + f[None, :],
            covered[:, None] & (f[None, :] < R),
            0,
        )
        acc += tl.dot(h, beta, input_precision="ieee")
    addr = (b[:, None] * J + node) * R + f[None, :]
    mask = (b[:, None] < B) & (f[None, :] < R)
    tl.store(C + addr, tl.load(C + addr, mask, 0) + acc, mask)


@tr.jit
def reconstruct(
    C,
    Phi,
    Banks,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    J: tl.constexpr,
    K: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    FF: tl.constexpr,
):
    b = tl.program_id(0) * BB + tl.arange(0, BB)
    s = tl.program_id(1) * BS + tl.arange(0, BS)
    f = tl.arange(0, FF)
    acc = tl.full((BB, BS), 0.0, tl.float32)
    for bank in range(K):
        S = tl.load(Banks + bank * 2)
        offset = tl.load(Banks + bank * 2 + 1)
        tile = s // S
        z = tl.load(
            C
            + (b[:, None, None] * J + offset + tile[None, :, None]) * R
            + f[None, None, :],
            (b[:, None, None] < B)
            & (s[None, :, None] < N // S * S)
            & (f[None, None, :] < R),
            0,
        )
        phi = tl.load(
            Phi + (bank * N + s[:, None]) * R + f[None, :],
            (s[:, None] < N) & (f[None, :] < R),
            0,
        )
        acc += tl.sum(z * phi[None, :, :], 2)
    addr = b[:, None] * N + s[None, :]
    mask = (b[:, None] < B) & (s[None, :] < N)
    tl.store(Y + addr, tl.load(Y + addr, mask, 0) + acc, mask)


@tr.jit
def deposit(
    H,
    P,
    Meta,
    Co,
    Ci,
    Norm,
    Banks,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    AXIS: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
):
    a, c = tl.program_id(0) // 4, tl.program_id(0) % 4
    b = tl.program_id(1) * BB + tl.arange(0, BB)
    if tl.load(Meta + (a * 4 + c) * 4 + 3) > 0:
        mu = tl.load(P + a * 4 + AXIS + 1).to(tl.float64)
        L = tl.load(Meta + (a * 4 + c) * 4)
        bank = tl.load(Meta + (a * 4 + c) * 4 + 2).to(tl.int32)
        lo, hi = _bounds(mu, L, N)
        left, right = lo, lo
        if bank >= 0:
            S = tl.load(Banks + bank * 2)
            left, right = (lo + S - 1) // S, hi // S
            if right > left:
                left, right = left * S, right * S
            else:
                left, right = lo, lo
        h = tl.load(H + (b * A + a) * 3, b < B, 0).to(tl.float64) * _alpha(P, Norm, a)
        for part in tl.static_range(2):
            begin = tl.where(part == 0, lo, right)
            end = tl.where(part == 0, left, hi)
            for start in range(begin, end, BS):
                s = start + tl.arange(0, BS)
                fv, _, _ = _component(s, mu, a, c, Meta, Co, Ci, M, False)
                tl.atomic_add(
                    Y + b[:, None] * N + s[None, :],
                    (h[:, None] * fv[None, :]).to(tl.float32),
                    (b[:, None] < B) & (s[None, :] < end),
                    sem="relaxed",
                )


@tr.jit
def deposit_cached(
    H,
    P,
    Meta,
    Co,
    Ci,
    Norm,
    Banks,
    Y,
    Native,
    Bounds,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    M: tl.constexpr,
    AXIS: tl.constexpr,
    BB: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
):
    a, c = tl.program_id(0) // 4, tl.program_id(0) % 4
    count = tl.load(Bounds + (a * 2 + AXIS) * 2 + 1)
    if count >= 0:
        if c == 0:
            b = tl.program_id(1) * BB + tl.arange(0, BB)
            lo = tl.load(Bounds + (a * 2 + AXIS) * 2)
            s = tl.arange(0, NC)
            f = tl.load(Native + ((a * 2 + AXIS) * 3) * NC + s, s < count, 0).to(
                tl.float64
            )
            h = tl.load(H + (b * A + a) * 3, b < B, 0).to(tl.float64) * _alpha(
                P, Norm, a
            )
            tl.atomic_add(
                Y + b[:, None] * N + lo + s[None, :],
                (h[:, None] * f[None, :]).to(tl.float32),
                (b[:, None] < B) & (s[None, :] < count),
                sem="relaxed",
            )
    else:
        deposit(H, P, Meta, Co, Ci, Norm, Banks, Y, B, N, A, M, AXIS, BB, BS)


@tr.jit
def vjp(H, G, P, Norm, Grad, B: tl.constexpr, A: tl.constexpr, BB: tl.constexpr):
    a = tl.program_id(0)
    b = tl.arange(0, BB)
    h = tl.load(H + (b * A + a) * 3, b < B, 0).to(tl.float64)
    hm = tl.load(H + (b * A + a) * 3 + 1, b < B, 0).to(tl.float64)
    hi = tl.load(H + (b * A + a) * 3 + 2, b < B, 0).to(tl.float64)
    g = tl.load(G + (b * A + a) * 3, b < B, 0).to(tl.float64)
    gm = tl.load(G + (b * A + a) * 3 + 1, b < B, 0).to(tl.float64)
    gi = tl.load(G + (b * A + a) * 3 + 2, b < B, 0).to(tl.float64)
    no = tl.load(Norm + a * 6)
    ni = tl.load(Norm + a * 6 + 3)
    raw = lib.sqrt(no * ni)
    d = tl.maximum(raw, 1.0e-6)
    amp = tl.load(P + a * 4).to(tl.float64)
    dot = tl.sum(h * g, 0)
    om = tl.where((raw > 1.0e-6) & (no > 0), tl.load(Norm + a * 6 + 1) / no, 0.0)
    im = tl.where((raw > 1.0e-6) & (ni > 0), tl.load(Norm + a * 6 + 4) / ni, 0.0)
    oi = tl.where((raw > 1.0e-6) & (no > 0), tl.load(Norm + a * 6 + 2) / no, 0.0)
    ii = tl.where((raw > 1.0e-6) & (ni > 0), tl.load(Norm + a * 6 + 5) / ni, 0.0)
    sig = 1 / (1 + lib.exp(-tl.load(P + a * 4 + 3).to(tl.float64)))
    tl.store(Grad + a * 4, dot / d)
    tl.store(Grad + a * 4 + 1, amp / d * (tl.sum(h * gm, 0) - dot * om))
    tl.store(Grad + a * 4 + 2, amp / d * (tl.sum(hm * g, 0) - dot * im))
    tl.store(
        Grad + a * 4 + 3,
        amp
        / d
        * (tl.sum(hi * g + h * gi, 0) - dot * (oi + ii))
        * 9.9
        * sig
        * (1 - sig),
    )
