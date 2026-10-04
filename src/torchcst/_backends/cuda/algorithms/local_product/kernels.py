"""Local H lives between two contractions; W is never constructed.

Saved mode uses the identical profiles but writes H[B,A] to global memory.
Both modes use full-domain normalization and its center derivatives.
"""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _raw(delta, inv):
    gap = tl.maximum(1.0 - delta * delta * inv, 0.0)
    return gap * gap * gap, 6.0 * delta * inv * gap * gap


@tr.jit
def _polar_atom(Source, a, Scalars):
    z0, z1 = tl.load(Source + 4 * a), tl.load(Source + 4 * a + 1)
    r2 = z0 * z0 + z1 * z1
    radius = libdevice.sqrt(tl.maximum(r2, 1.1754943508222875e-38))
    amplitude = tl.div_rn(tl.load(Scalars[0]).to(tl.float32) * z0, radius)
    alpha = tl.minimum(tl.maximum(tl.div_rn(r2 - 1.0, 3.0), 0.0), 1.0)
    minimum = tl.load(Scalars[5]).to(tl.float32)
    birth = tl.load(Scalars[6]).to(tl.float32)
    maximum = tl.load(Scalars[7]).to(tl.float32)
    kappa = tl.load(Scalars[2]).to(tl.float32)
    ratio = tl.div_rn(amplitude, tl.load(Scalars[1]).to(tl.float32))
    x = ratio * ratio
    upper_x = libdevice.pow(x, tl.load(Scalars[4]).to(tl.float32))
    upper = minimum + tl.div_rn((maximum - minimum) * kappa, kappa + upper_x)
    upper = tl.maximum(upper, tl.load(Scalars[8]).to(tl.float32))
    lower = minimum + tl.div_rn(
        birth - minimum, 1.0 + tl.load(Scalars[3]).to(tl.float32) * x
    )
    upper = tl.maximum(upper, lower)
    sigma = libdevice.exp(
        (1.0 - alpha) * libdevice.log(lower) + alpha * libdevice.log(upper)
    )
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    inverse = tl.div_rn(1.0, sigma)
    return amplitude, inverse * inverse


@tr.jit
def prepare(
    Q,
    P,
    A: tl.constexpr,
    KI: tl.constexpr,
    NO: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    PK: tl.constexpr,
    PN: tl.constexpr,
    BOUNDS: tl.constexpr = False,
    POLAR: tl.constexpr = False,
    Scalars=(),
):
    a = tl.program_id(0)
    if POLAR:
        amp, inv = _polar_atom(Q, a, Scalars)
    else:
        amp = tl.load(Q + a * 4)
        inv = tl.load(Q + a * 4 + 1)
    ci = tl.load(Q + a * 4 + 2)
    co = tl.load(Q + a * 4 + 3)
    j, i = tl.arange(0, PK), tl.arange(0, PN)
    v, dv = _raw(OI + j * S - ci, inv)
    u, du = _raw(OO + i * S - co, inv)
    v, dv = tl.where(j < KI, v, 0.0), tl.where(j < KI, dv, 0.0)
    u, du = tl.where(i < NO, u, 0.0), tl.where(i < NO, du, 0.0)
    nv, nu = tl.sqrt(tl.sum(v * v, 0)), tl.sqrt(tl.sum(u * u, 0))
    sv, su = tl.maximum(nv, 1e-6), tl.maximum(nu, 1e-6)
    gv = tl.where(nv >= 1e-6, tl.sum(v * dv, 0) / (sv * sv), 0.0)
    gu = tl.where(nu >= 1e-6, tl.sum(u * du, 0) / (su * su), 0.0)
    flags = ((nv >= 1e-6) & (tl.sum((v > 0).to(tl.int32), 0) == 1)).to(tl.int32) | (
        ((nu >= 1e-6) & (tl.sum((u > 0).to(tl.int32), 0) == 1)).to(tl.int32) << 1
    )
    tl.store(P + a, amp)
    tl.store(P + A + a, inv)
    tl.store(P + 2 * A + a, ci)
    tl.store(P + 3 * A + a, co)
    tl.store(P + 4 * A + a, sv)
    tl.store(P + 5 * A + a, su)
    tl.store(P + 6 * A + a, gv)
    tl.store(P + 7 * A + a, gu)
    tl.store(P + 8 * A + a, flags.to(tl.float32))
    if BOUNDS:
        # Full-domain support, independently for both centers. The same sigma
        # need not imply equal counts near a boundary or at different grid phases.
        tl.store(P + 9 * A + a, tl.min(tl.where(v > 0, j, KI), 0).to(tl.float32))
        tl.store(P + 10 * A + a, tl.max(tl.where(v > 0, j + 1, 0), 0).to(tl.float32))
        tl.store(P + 11 * A + a, tl.min(tl.where(u > 0, i, NO), 0).to(tl.float32))
        tl.store(P + 12 * A + a, tl.max(tl.where(u > 0, i + 1, 0), 0).to(tl.float32))


@tr.jit
def _factor(
    P, a, points, A: tl.constexpr, OUT: tl.constexpr, S: tl.constexpr, O: tl.constexpr
):
    valid = a < A
    inv = tl.load(P + A + a, valid, 0.0)
    c = tl.load(P + (3 if OUT else 2) * A + a, valid, 0.0)
    norm = tl.load(P + (5 if OUT else 4) * A + a, valid, 1.0)
    gamma = tl.load(P + (7 if OUT else 6) * A + a, valid, 0.0)
    flag = tl.load(P + 8 * A + a, valid, 0.0).to(tl.int32)
    singleton = ((flag >> (1 if OUT else 0)) & 1) != 0
    raw, dc = _raw(O + points[:, None] * S - c[None, :], inv[None, :])
    f = raw / norm[None, :]
    d = dc / norm[None, :] - f * gamma[None, :]
    d = tl.where(singleton[None, :], 0.0, d)
    return tl.where(valid[None, :], f, 0.0), tl.where(valid[None, :], d, 0.0)


@tr.jit
def _interval(
    P, a, A: tl.constexpr, OUT: tl.constexpr, START: tl.constexpr, COUNT: tl.constexpr
):
    lo = tl.load(P + (11 if OUT else 9) * A + a, a < A, 0.0).to(tl.int32)
    hi = tl.load(P + (12 if OUT else 10) * A + a, a < A, 0.0).to(tl.int32)
    lo = tl.minimum(tl.maximum(lo - START, 0), COUNT)
    hi = tl.minimum(tl.maximum(hi - START, 0), COUNT)
    return lo, hi, tl.maximum(hi - lo, 0)


@tr.jit
def _site_factor(
    P, a, points, A: tl.constexpr, OUT: tl.constexpr, S: tl.constexpr, O: tl.constexpr
):
    inv = tl.load(P + A + a, a < A, 0.0)
    c = tl.load(P + (3 if OUT else 2) * A + a, a < A, 0.0)
    norm = tl.load(P + (5 if OUT else 4) * A + a, a < A, 1.0)
    gamma = tl.load(P + (7 if OUT else 6) * A + a, a < A, 0.0)
    flag = tl.load(P + 8 * A + a, a < A, 0.0).to(tl.int32)
    raw, dc = _raw(O + points * S - c, inv)
    f = raw / norm
    d = dc / norm - f * gamma
    d = tl.where(((flag >> (1 if OUT else 0)) & 1) != 0, 0.0, d)
    return f, d


@tr.jit
def _support_contract(
    X,
    P,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    OUT: tl.constexpr,
    S: tl.constexpr,
    O: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    NARROW_ONLY: tl.constexpr = False,
    RHO: tl.constexpr = 4.0,
    THREE_BAND: tl.constexpr = False,
    Enabled=None,
):
    lo, _hi, width = _interval(P, a, A, OUT, START, K)
    if Enabled is not None:
        width = tl.where(Enabled, width, 0)
    if NARROW_ONLY:
        width = tl.where(_wide(P, a, A, S, RHO, THREE_BAND), 0, width)
    h, dh = tl.full((BB, BA), 0.0, tl.float32), tl.full((BB, BA), 0.0, tl.float32)
    if THREE_BAND:
        inv = tl.load(P + A + a, a < A, 0.0)
        # rho<1 is at most two positive sites, not a one-hot certificate.
        # Keep an actual-count guard and the original full-domain factor/floor.
        fast = (a < A) & (inv > 1.0 / (S * S)) & (width <= 2)
        for offset in tl.static_range(2):
            j = lo + offset
            live = fast & (offset < width) & (j < K)
            x = tl.load(
                X + b[:, None] * K + j[None, :],
                (b[:, None] < B) & live[None, :],
                0.0,
            )
            f, dc = _site_factor(P, a, START + j, A, OUT, S, O)
            h += x * f[None, :]
            dh += x * dc[None, :]
        width = tl.where(fast, 0, width)
    for offset in range(tl.max(width, 0)):
        j = lo + offset
        live = (a < A) & (offset < width) & (j < K)
        x = tl.load(
            X + b[:, None] * K + j[None, :], (b[:, None] < B) & live[None, :], 0.0
        )
        f, dc = _site_factor(P, a, START + j, A, OUT, S, O)
        h += x * f[None, :]
        dh += x * dc[None, :]
    return h, dh


@tr.jit
def _matrix_contract(
    X,
    P,
    a,
    b,
    A: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    OUT: tl.constexpr,
    S: tl.constexpr,
    O: tl.constexpr,
    BK: tl.constexpr,
    WIDE_ONLY: tl.constexpr = False,
    RHO: tl.constexpr = 4.0,
    THREE_BAND: tl.constexpr = False,
    Enabled=None,
):
    j = tl.arange(0, BK)
    x = tl.load(
        X + b[:, None] * K + j[None, :], (b[:, None] < B) & (j[None, :] < K), 0.0
    )
    v, dv = _factor(P, a, START + j, A, OUT, S, O)
    v, dv = tl.where(j[:, None] < K, v, 0.0), tl.where(j[:, None] < K, dv, 0.0)
    if WIDE_ONLY:
        wide = _wide(P, a, A, S, RHO, THREE_BAND)
        v, dv = tl.where(wide[None, :], v, 0.0), tl.where(wide[None, :], dv, 0.0)
    if Enabled is not None:
        v, dv = tl.where(Enabled[None, :], v, 0.0), tl.where(Enabled[None, :], dv, 0.0)
    return tl.dot(x, v, input_precision="ieee"), tl.dot(x, dv, input_precision="ieee")


@tr.jit
def _wide(
    P,
    a,
    A: tl.constexpr,
    S: tl.constexpr,
    RHO: tl.constexpr,
    INCLUSIVE: tl.constexpr = False,
):
    # rho > RHO, using the precision prepared from this forward's live sigma.
    inv = tl.load(P + A + a, a < A, 0.0)
    if INCLUSIVE:
        return (a < A) & (inv <= 1.0 / ((S * RHO) * (S * RHO)))
    return (a < A) & (inv < 1.0 / ((S * RHO) * (S * RHO)))


@tr.jit
def _singletons(P, a, A: tl.constexpr):
    # Both factors have exactly one full-domain positive site and live norms.
    flags = tl.load(P + 8 * A + a, a < A, 0.0).to(tl.int32)
    return (a < A) & (flags == 3)


@tr.jit
def fused(
    X,
    P,
    Y,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    SWAP: tl.constexpr,
    SPARSE: tl.constexpr = False,
    LIMIT: tl.constexpr = 8,
    HYBRID: tl.constexpr = False,
    H=None,
    RHO: tl.constexpr = 4.0,
    SUPPORT_ONLY: tl.constexpr = False,
    THREE_BAND: tl.constexpr = False,
    SINGLETON_FAST: tl.constexpr = False,
):
    b = tl.program_id(0) * BM + tl.arange(0, BM)
    j, i = tl.arange(0, BK), tl.arange(0, BN)
    if SINGLETON_FAST:
        i += tl.program_id(1) * BN
    if not SPARSE:
        x = tl.load(
            X + b[:, None] * K + j[None, :], (b[:, None] < B) & (j[None, :] < K), 0.0
        )
    y = tl.full((BM, BN), 0.0, tl.float32)
    for a0 in range(0, A, BA):
        a = a0 + tl.arange(0, BA)
        if SINGLETON_FAST:
            scalar = _singletons(P, a, A)
            vlo, _vhi, vw = _interval(P, a, A, SWAP, JS, K)
            ulo, uhi, uw = _interval(P, a, A, not SWAP, IS, N)
            tile = tl.program_id(1) * BN
            recipient = (ulo >= tile) & (ulo < tile + BN)
            live = scalar & (vw == 1) & (uw == 1) & recipient
            direct = tl.load(
                X + b[:, None] * K + vlo[None, :],
                (b[:, None] < B) & live[None, :],
                0.0,
            )
            amp_single = tl.load(P + a, live, 0.0)
            targets = ((i[:, None] == ulo[None, :]) & live[None, :]).to(tl.float32)
            y = tl.dot(
                direct * amp_single[None, :],
                tl.trans(targets),
                y,
                input_precision="ieee",
            )
            enabled = (a < A) & ~scalar & (ulo < tile + BN) & (uhi > tile)
        else:
            enabled = a < A
        work = tl.sum(enabled.to(tl.int32), 0) if SINGLETON_FAST else 1
        if work > 0:
            y = _fused_general_block(
                X,
                P,
                H,
                a,
                b,
                j,
                i,
                y,
                enabled,
                B,
                K,
                N,
                A,
                S,
                OI,
                OO,
                JS,
                IS,
                BK,
                BM,
                BA,
                SWAP,
                SPARSE,
                LIMIT,
                HYBRID,
                RHO,
                SUPPORT_ONLY,
                THREE_BAND,
                DenseX=x if not SPARSE else None,
            )
    tl.store(Y + b[:, None] * N + i[None, :], y, (b[:, None] < B) & (i[None, :] < N))


@tr.jit
def _fused_general_block(
    X,
    P,
    H,
    a,
    b,
    j,
    i,
    y,
    enabled,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    SWAP: tl.constexpr,
    SPARSE: tl.constexpr,
    LIMIT: tl.constexpr,
    HYBRID: tl.constexpr,
    RHO: tl.constexpr,
    SUPPORT_ONLY: tl.constexpr,
    THREE_BAND: tl.constexpr,
    DenseX=None,
):
    if not SPARSE:
        x = DenseX
    u, _du_full = _factor(P, a, IS + i, A, not SWAP, S, OO)
    u = tl.where((i[:, None] < N) & enabled[None, :], u, 0.0)
    amp = tl.load(P + a, a < A, 0.0)
    if HYBRID:
        wide = _wide(P, a, A, S, RHO, THREE_BAND) & enabled
        narrow = enabled & ~wide
        h = tl.full((BM, BA), 0.0, tl.float32)
        if tl.sum(narrow.to(tl.int32), 0) > 0:
            if SPARSE:
                h, _dh_local_support = _support_contract(
                    X,
                    P,
                    a,
                    b,
                    A,
                    B,
                    K,
                    JS,
                    False,
                    S,
                    OI,
                    BM,
                    BA,
                    True,
                    RHO,
                    THREE_BAND,
                    Enabled=enabled,
                )
            else:
                v_local, _dv_local_matrix = _factor(P, a, JS + j, A, False, S, OI)
                v_local = tl.where((j[:, None] < K) & narrow[None, :], v_local, 0.0)
                h = tl.dot(x, v_local, input_precision="ieee")
        if tl.sum(wide.to(tl.int32), 0) > 0:
            stored = tl.load(
                H + b[:, None] * A + a[None, :],
                (b[:, None] < B) & wide[None, :],
                0.0,
            )
            h = tl.where(wide[None, :], stored, h)
    elif SPARSE:
        _lo, _hi, width = _interval(P, a, A, SWAP, JS, K)
        if SUPPORT_ONLY or tl.max(width, 0) <= LIMIT:
            h, _dh_support = _support_contract(
                X, P, a, b, A, B, K, JS, SWAP, S, OI, BM, BA, Enabled=enabled
            )
        else:
            h, _dh_matrix = _matrix_contract(
                X, P, a, b, A, B, K, JS, SWAP, S, OI, BK, Enabled=enabled
            )
    else:
        v, _dv_full = _factor(P, a, JS + j, A, SWAP, S, OI)
        v = tl.where((j[:, None] < K) & enabled[None, :], v, 0.0)
        h = tl.dot(x, v, input_precision="ieee")
    return tl.dot(h * amp[None, :], tl.trans(u), y, input_precision="ieee")


@tr.jit
def save_h(
    X,
    P,
    H,
    B: tl.constexpr,
    K: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    JS: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    HYBRID: tl.constexpr = False,
    RHO: tl.constexpr = 4.0,
    SUPPORT_ONLY: tl.constexpr = False,
    THREE_BAND: tl.constexpr = False,
    SINGLETON_FAST: tl.constexpr = False,
):
    b = tl.program_id(0) * BM + tl.arange(0, BM)
    a = tl.program_id(1) * BA + tl.arange(0, BA)
    j = tl.arange(0, BK)
    save = _wide(P, a, A, S, RHO, THREE_BAND) if HYBRID else (a < A)
    if SINGLETON_FAST:
        save &= ~_singletons(P, a, A)
    if not HYBRID or tl.sum(save.to(tl.int32), 0) > 0:
        if SUPPORT_ONLY:
            h, _dh_saved = _support_contract(
                X, P, a, b, A, B, K, JS, False, S, OI, BM, BA
            )
        else:
            x = tl.load(
                X + b[:, None] * K + j[None, :],
                (b[:, None] < B) & (j[None, :] < K),
                0.0,
            )
            v, _dv_saved = _factor(P, a, JS + j, A, False, S, OI)
            v = tl.where((j[:, None] < K) & save[None, :], v, 0.0)
            h = tl.dot(x, v, input_precision="ieee")
        tl.store(H + b[:, None] * A + a[None, :], h, (b[:, None] < B) & save[None, :])


@tr.jit
def from_h(
    H,
    P,
    Y,
    B: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OO: tl.constexpr,
    IS: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
):
    b = tl.program_id(0) * BM + tl.arange(0, BM)
    i = tl.arange(0, BN)
    y = tl.full((BM, BN), 0.0, tl.float32)
    for a0 in range(0, A, BA):
        a = a0 + tl.arange(0, BA)
        h = tl.load(
            H + b[:, None] * A + a[None, :], (b[:, None] < B) & (a[None, :] < A), 0.0
        )
        u, _ = _factor(P, a, IS + i, A, True, S, OO)
        u = tl.where(i[:, None] < N, u, 0.0)
        amp = tl.load(P + a, a < A, 0.0)
        y = tl.dot(h * amp[None, :], tl.trans(u), y, input_precision="ieee")
    tl.store(Y + b[:, None] * N + i[None, :], y, (b[:, None] < B) & (i[None, :] < N))


@tr.jit
def _param_sums(
    X,
    DY,
    P,
    H,
    a,
    b,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    SAVED: tl.constexpr,
    SPARSE: tl.constexpr,
    HYBRID: tl.constexpr,
    RHO: tl.constexpr,
    LIMIT: tl.constexpr,
    SUPPORT_ONLY: tl.constexpr,
    THREE_BAND: tl.constexpr,
    SINGLETON_FAST: tl.constexpr,
):
    j, i = tl.arange(0, BK), tl.arange(0, BN)
    enabled = (a < A) & ~_singletons(P, a, A) if SINGLETON_FAST else (a < A)
    if HYBRID and SPARSE:
        h, dh = _support_contract(
            X,
            P,
            a,
            b,
            A,
            B,
            K,
            JS,
            False,
            S,
            OI,
            BB,
            BA,
            True,
            RHO,
            THREE_BAND,
            Enabled=enabled,
        )
        g, dg = _support_contract(
            DY,
            P,
            a,
            b,
            A,
            B,
            N,
            IS,
            True,
            S,
            OO,
            BB,
            BA,
            True,
            RHO,
            THREE_BAND,
            Enabled=enabled,
        )
        wide = _wide(P, a, A, S, RHO, THREE_BAND) & enabled
        if tl.sum(wide.to(tl.int32), 0) > 0:
            _h_wide_unused, dh_wide = _matrix_contract(
                X,
                P,
                a,
                b,
                A,
                B,
                K,
                JS,
                False,
                S,
                OI,
                BK,
                True,
                RHO,
                THREE_BAND,
                Enabled=enabled,
            )
            g_wide, dg_wide = _matrix_contract(
                DY,
                P,
                a,
                b,
                A,
                B,
                N,
                IS,
                True,
                S,
                OO,
                BN,
                True,
                RHO,
                THREE_BAND,
                Enabled=enabled,
            )
            stored_wide_h = tl.load(
                H + b[:, None] * A + a[None, :],
                (b[:, None] < B) & wide[None, :],
                0.0,
            )
            h, dh = (
                tl.where(wide[None, :], stored_wide_h, h),
                tl.where(wide[None, :], dh_wide, dh),
            )
            g, dg = (
                tl.where(wide[None, :], g_wide, g),
                tl.where(wide[None, :], dg_wide, dg),
            )
    elif SPARSE:
        _vlo, _vhi, vw = _interval(P, a, A, False, JS, K)
        _ulo, _uhi, uw = _interval(P, a, A, True, IS, N)
        if SUPPORT_ONLY or tl.maximum(tl.max(vw, 0), tl.max(uw, 0)) <= LIMIT:
            h, dh = _support_contract(X, P, a, b, A, B, K, JS, False, S, OI, BB, BA)
            g, dg = _support_contract(DY, P, a, b, A, B, N, IS, True, S, OO, BB, BA)
        else:
            h, dh = _matrix_contract(X, P, a, b, A, B, K, JS, False, S, OI, BK)
            g, dg = _matrix_contract(DY, P, a, b, A, B, N, IS, True, S, OO, BN)
        if SAVED:
            h = tl.load(
                H + b[:, None] * A + a[None, :],
                (b[:, None] < B) & (a[None, :] < A),
                0.0,
            )
    else:
        x = tl.load(
            X + b[:, None] * K + j[None, :], (b[:, None] < B) & (j[None, :] < K), 0.0
        )
        dy = tl.load(
            DY + b[:, None] * N + i[None, :], (b[:, None] < B) & (i[None, :] < N), 0.0
        )
        v, dv = _factor(P, a, JS + j, A, False, S, OI)
        u, du = _factor(P, a, IS + i, A, True, S, OO)
        v, dv = tl.where(j[:, None] < K, v, 0.0), tl.where(j[:, None] < K, dv, 0.0)
        u, du = tl.where(i[:, None] < N, u, 0.0), tl.where(i[:, None] < N, du, 0.0)
        if SAVED:
            h = tl.load(
                H + b[:, None] * A + a[None, :],
                (b[:, None] < B) & (a[None, :] < A),
                0.0,
            )
        elif HYBRID:
            wide = _wide(P, a, A, S, RHO)
            narrow = (a < A) & ~wide
            h = tl.full((BB, BA), 0.0, tl.float32)
            if tl.sum(narrow.to(tl.int32), 0) > 0:
                h = tl.dot(x, tl.where(narrow[None, :], v, 0.0), input_precision="ieee")
            if tl.sum(wide.to(tl.int32), 0) > 0:
                stored_h = tl.load(
                    H + b[:, None] * A + a[None, :],
                    (b[:, None] < B) & wide[None, :],
                    0.0,
                )
                h = tl.where(wide[None, :], stored_h, h)
        else:
            h = tl.dot(x, v, input_precision="ieee")
        g = tl.dot(dy, u, input_precision="ieee")
        dh = tl.dot(x, dv, input_precision="ieee")
        dg = tl.dot(dy, du, input_precision="ieee")
    amp = tl.load(P + a, a < A, 0.0)
    da = tl.sum(h * g, 0)
    dci = amp * tl.sum(dh * g, 0)
    dco = amp * tl.sum(h * dg, 0)
    return da, dci, dco


@tr.jit
def param_vjp(
    X,
    DY,
    P,
    H,
    DQ,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    JS: tl.constexpr,
    IS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    SAVED: tl.constexpr,
    SPARSE: tl.constexpr = False,
    LIMIT: tl.constexpr = 8,
    POLAR: tl.constexpr = False,
    Source=None,
    AmplitudeMax=None,
    HYBRID: tl.constexpr = False,
    RHO: tl.constexpr = 4.0,
    SUPPORT_ONLY: tl.constexpr = False,
    THREE_BAND: tl.constexpr = False,
    SINGLETON_FAST: tl.constexpr = False,
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    da = tl.full((BA,), 0.0, tl.float32)
    dci = tl.full((BA,), 0.0, tl.float32)
    dco = tl.full((BA,), 0.0, tl.float32)
    for b0 in range(0, B, BB):
        b = b0 + tl.arange(0, BB)
        partial_a = tl.full((BA,), 0.0, tl.float32)
        partial_i = tl.full((BA,), 0.0, tl.float32)
        partial_o = tl.full((BA,), 0.0, tl.float32)
        if SINGLETON_FAST:
            scalar = _singletons(P, a, A)
            general_count = tl.sum(((a < A) & ~scalar).to(tl.int32), 0)
        else:
            general_count = 1
        if general_count > 0:
            partial_a, partial_i, partial_o = _param_sums(
                X,
                DY,
                P,
                H,
                a,
                b,
                B,
                K,
                N,
                A,
                S,
                OI,
                OO,
                JS,
                IS,
                BK,
                BN,
                BB,
                BA,
                SAVED,
                SPARSE,
                HYBRID,
                RHO,
                LIMIT,
                SUPPORT_ONLY,
                THREE_BAND,
                SINGLETON_FAST,
            )
        if SINGLETON_FAST:
            vlo, _vhi, vw = _interval(P, a, A, False, JS, K)
            ulo, _uhi, uw = _interval(P, a, A, True, IS, N)
            live = scalar & (vw == 1) & (uw == 1)
            sx = tl.load(
                X + b[:, None] * K + vlo[None, :], (b[:, None] < B) & live[None, :], 0.0
            )
            sy = tl.load(
                DY + b[:, None] * N + ulo[None, :],
                (b[:, None] < B) & live[None, :],
                0.0,
            )
            partial_a = tl.where(scalar, tl.sum(sx * sy, 0), partial_a)
            partial_i = tl.where(scalar, 0.0, partial_i)
            partial_o = tl.where(scalar, 0.0, partial_o)
        da += partial_a
        dci += partial_i
        dco += partial_o
    if POLAR:
        z0 = tl.load(Source + 4 * a, a < A, 0.0)
        z1 = tl.load(Source + 4 * a + 1, a < A, 0.0)
        r2 = z0 * z0 + z1 * z1
        safe = tl.maximum(r2, 1.1754943508222875e-38)
        inverse = tl.div_rn(1.0, libdevice.sqrt(safe))
        active = r2 >= 1.1754943508222875e-38
        scale = tl.load(AmplitudeMax).to(tl.float32) * inverse
        d0 = da * scale * (1.0 - tl.where(active, tl.div_rn(z0 * z0, safe), 0.0))
        d1 = -da * scale * tl.where(active, tl.div_rn(z0 * z1, safe), 0.0)
        tl.store(DQ + 4 * a, d0, a < A)
        tl.store(DQ + 4 * a + 1, d1, a < A)
    else:
        tl.store(DQ + 4 * a, da, a < A)
        tl.store(DQ + 4 * a + 1, 0.0, a < A)
    tl.store(DQ + 4 * a + 2, dci, a < A)
    tl.store(DQ + 4 * a + 3, dco, a < A)
