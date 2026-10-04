"""Local H lives between two contractions; W is never constructed.

Saved mode uses the identical profiles but writes H[B,A] to global memory.
Both modes use full-domain normalization and its center derivatives.
"""

import triton as tr
import triton.language as tl


@tr.jit
def _raw(delta, inv):
    gap = tl.maximum(1.0 - delta * delta * inv, 0.0)
    return gap * gap * gap, 6.0 * delta * inv * gap * gap


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
):
    a = tl.program_id(0)
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
):
    b = tl.program_id(0) * BM + tl.arange(0, BM)
    j, i = tl.arange(0, BK), tl.arange(0, BN)
    x = tl.load(
        X + b[:, None] * K + j[None, :], (b[:, None] < B) & (j[None, :] < K), 0.0
    )
    y = tl.full((BM, BN), 0.0, tl.float32)
    for a0 in range(0, A, BA):
        a = a0 + tl.arange(0, BA)
        v, _ = _factor(P, a, JS + j, A, SWAP, S, OI)
        u, _ = _factor(P, a, IS + i, A, not SWAP, S, OO)
        v, u = tl.where(j[:, None] < K, v, 0.0), tl.where(i[:, None] < N, u, 0.0)
        amp = tl.load(P + a, a < A, 0.0)
        h = tl.dot(x, v, input_precision="ieee")
        y = tl.dot(h * amp[None, :], tl.trans(u), y, input_precision="ieee")
    tl.store(Y + b[:, None] * N + i[None, :], y, (b[:, None] < B) & (i[None, :] < N))


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
):
    b = tl.program_id(0) * BM + tl.arange(0, BM)
    a = tl.program_id(1) * BA + tl.arange(0, BA)
    j = tl.arange(0, BK)
    x = tl.load(
        X + b[:, None] * K + j[None, :], (b[:, None] < B) & (j[None, :] < K), 0.0
    )
    v, _ = _factor(P, a, JS + j, A, False, S, OI)
    v = tl.where(j[:, None] < K, v, 0.0)
    h = tl.dot(x, v, input_precision="ieee")
    tl.store(H + b[:, None] * A + a[None, :], h, (b[:, None] < B) & (a[None, :] < A))


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
):
    a = tl.program_id(0) * BA + tl.arange(0, BA)
    b, j, i = tl.arange(0, BB), tl.arange(0, BK), tl.arange(0, BN)
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
            H + b[:, None] * A + a[None, :], (b[:, None] < B) & (a[None, :] < A), 0.0
        )
    else:
        h = tl.dot(x, v, input_precision="ieee")
    g = tl.dot(dy, u, input_precision="ieee")
    dh = tl.dot(x, dv, input_precision="ieee")
    dg = tl.dot(dy, du, input_precision="ieee")
    amp = tl.load(P + a, a < A, 0.0)
    da = tl.sum(h * g, 0)
    dci = amp * tl.sum(dh * g, 0)
    dco = amp * tl.sum(h * dg, 0)
    tl.store(DQ + 4 * a, da, a < A)
    tl.store(DQ + 4 * a + 1, 0.0, a < A)
    tl.store(DQ + 4 * a + 2, dci, a < A)
    tl.store(DQ + 4 * a + 3, dco, a < A)
