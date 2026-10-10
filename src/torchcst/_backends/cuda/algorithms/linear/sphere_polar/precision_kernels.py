"""Sensitive-atom CUDA Core contractions from physical FP64 profile derivatives.

Only immutable snapshots are read. Profiles and already-normalized intrinsic
derivatives are cast to FP32 before contractions; no late norm cancellation.
"""

import triton as tr
import triton.language as tl

from .precision_prepare_kernels import _f64, physical_raw


@tr.jit
def _block(
    S,
    Q,
    J,
    P,
    Index,
    Norm,
    Count,
    Moment,
    a,
    base,
    N: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
):
    off = base + tl.arange(0, T)
    count = tl.load(Count + a)
    full = count > CAP
    valid = tl.where(full, off < N, off < count)
    stored = tl.load(Index + a * CAP + off, ~full & (off < count) & (off < CAP), 0).to(
        tl.int32
    )
    idx = tl.where(full, off, stored)
    gap, d0, d1, d2, precision = physical_raw(S, Q, P, a, idx, N)
    gap = tl.where(valid, gap, 0.0)
    norm = tl.load(Norm + a)
    # Keep the same FP64 floor in the denominator and derivative branch.
    # Implicit FP32 constexpr promotion can flip the exact-equality branch.
    floor = _f64(FLOOR)
    denominator = tl.maximum(norm, floor)
    phi = gap * gap * gap / denominator
    coefficient = 6.0 * precision * gap * gap / denominator
    project = tl.where(norm >= floor, phi, 0.0)
    t0 = coefficient * d0 - project * tl.load(Moment + 3 * a)
    t1 = coefficient * d1 - project * tl.load(Moment + 3 * a + 1)
    t2 = coefficient * d2 - project * tl.load(Moment + 3 * a + 2)
    g0 = (
        t0 * tl.load(J + 6 * a)
        + t1 * tl.load(J + 6 * a + 2)
        + t2 * tl.load(J + 6 * a + 4)
    )
    g1 = (
        t0 * tl.load(J + 6 * a + 1)
        + t1 * tl.load(J + 6 * a + 3)
        + t2 * tl.load(J + 6 * a + 5)
    )
    # Exact analytic identity, never used under the normalization floor.
    singleton = (count == 1) & (norm >= floor)
    g0 = tl.where(singleton | ~valid, 0.0, g0).to(tl.float32)
    g1 = tl.where(singleton | ~valid, 0.0, g1).to(tl.float32)
    return idx, valid, phi.to(tl.float32), g0, g1


@tr.jit
def forward(
    X,
    SI,
    QI,
    JI,
    PI,
    II,
    NI,
    CI,
    MI,
    SO,
    QO,
    JO,
    PO,
    IO,
    NO,
    CO,
    MO,
    Amp,
    H,
    Y,
    Sensitive,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    PB: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(Sensitive + a):
        batch = tl.arange(0, PB)
        ci, co = tl.load(CI + a), tl.load(CO + a)
        ni = tl.where(ci > CAP, IN, ci)
        no = tl.where(co > CAP, OUT, co)
        h = tl.full((PB,), 0.0, tl.float32)
        for base in range(0, ni, T):
            idx, valid, phi, _, _ = _block(
                SI, QI, JI, PI, II, NI, CI, MI, a, base, IN, CAP, T, FLOOR_I
            )
            xx = tl.load(
                X + batch[:, None] * IN + idx[None, :],
                (batch[:, None] < B) & valid[None, :],
                0,
            )
            h += tl.sum(xx * phi[None, :], 1)
        if SAVE_H:
            tl.store(H + a * B + batch, h, batch < B)
        scaled = h * tl.load(Amp + a)
        for base in range(0, no, T):
            idx, valid, phi, _, _ = _block(
                SO, QO, JO, PO, IO, NO, CO, MO, a, base, OUT, CAP, T, FLOOR_O
            )
            tl.atomic_add(
                Y + batch[:, None] * OUT + idx[None, :],
                scaled[:, None] * phi[None, :],
                (batch[:, None] < B) & valid[None, :] & (phi[None, :] != 0),
                sem="relaxed",
            )


@tr.jit
def backward(
    X,
    DY,
    SI,
    QI,
    JI,
    PI,
    II,
    NI,
    CI,
    MI,
    SO,
    QO,
    JO,
    PO,
    IO,
    NO,
    CO,
    MO,
    Amp,
    Damp,
    H,
    DX,
    DP,
    Sensitive,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    PB: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(Sensitive + a):
        batch = tl.arange(0, PB)
        ci, co = tl.load(CI + a), tl.load(CO + a)
        ni, no = tl.where(ci > CAP, IN, ci), tl.where(co > CAP, OUT, co)
        g = tl.full((PB,), 0.0, tl.float32)
        go0, go1 = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        if NEED_P:
            h = tl.load(H + a * B + batch, batch < B, 0)
        for base in range(0, no, T):
            idx, valid, phi, d0, d1 = _block(
                SO, QO, JO, PO, IO, NO, CO, MO, a, base, OUT, CAP, T, FLOOR_O
            )
            dy = tl.load(
                DY + batch[:, None] * OUT + idx[None, :],
                (batch[:, None] < B) & valid[None, :],
                0,
            )
            g += tl.sum(dy * phi[None, :], 1)
            if NEED_P:
                weight = tl.sum(dy * h[:, None], 0)
                go0 += tl.sum(weight * d0, 0)
                go1 += tl.sum(weight * d1, 0)
        amp = tl.load(Amp + a)
        gi0, gi1 = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        for base in range(0, ni, T):
            idx, valid, phi, d0, d1 = _block(
                SI, QI, JI, PI, II, NI, CI, MI, a, base, IN, CAP, T, FLOOR_I
            )
            if NEED_X:
                tl.atomic_add(
                    DX + batch[:, None] * IN + idx[None, :],
                    amp * g[:, None] * phi[None, :],
                    (batch[:, None] < B) & valid[None, :] & (phi[None, :] != 0),
                    sem="relaxed",
                )
            if NEED_P:
                xx = tl.load(
                    X + batch[:, None] * IN + idx[None, :],
                    (batch[:, None] < B) & valid[None, :],
                    0,
                )
                weight = tl.sum(xx * g[:, None], 0)
                gi0 += tl.sum(weight * d0, 0)
                gi1 += tl.sum(weight * d1, 0)
        if NEED_P:
            dot = tl.sum(h * g, 0)
            tl.store(DP + a * 6, dot * tl.load(Damp + a * 2))
            tl.store(DP + a * 6 + 1, dot * tl.load(Damp + a * 2 + 1))
            tl.store(DP + a * 6 + 2, amp * gi0)
            tl.store(DP + a * 6 + 3, amp * gi1)
            tl.store(DP + a * 6 + 4, amp * go0)
            tl.store(DP + a * 6 + 5, amp * go1)


@tr.jit
def assemble(
    SI,
    QI,
    JI,
    PI,
    II,
    NI,
    CI,
    MI,
    SO,
    QO,
    JO,
    PO,
    IO,
    NO,
    CO,
    MO,
    Amp,
    W,
    Sensitive,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(Sensitive + a):
        ci, co = tl.load(CI + a), tl.load(CO + a)
        ni, no = tl.where(ci > CAP, IN, ci), tl.where(co > CAP, OUT, co)
        amp = tl.load(Amp + a)
        for ib in range(0, ni, T):
            ii, vi, fi, _, _ = _block(
                SI, QI, JI, PI, II, NI, CI, MI, a, ib, IN, CAP, T, FLOOR_I
            )
            for ob in range(0, no, T):
                io, vo, fo, _, _ = _block(
                    SO, QO, JO, PO, IO, NO, CO, MO, a, ob, OUT, CAP, T, FLOOR_O
                )
                tl.atomic_add(
                    W + io[:, None] * IN + ii[None, :],
                    amp * fo[:, None] * fi[None, :],
                    vo[:, None] & vi[None, :] & (fo[:, None] != 0) & (fi[None, :] != 0),
                    sem="relaxed",
                )


@tr.jit
def vjp(
    SI,
    QI,
    JI,
    PI,
    II,
    NI,
    CI,
    MI,
    SO,
    QO,
    JO,
    PO,
    IO,
    NO,
    CO,
    MO,
    Amp,
    Damp,
    DW,
    DP,
    Sensitive,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(Sensitive + a):
        ci, co = tl.load(CI + a), tl.load(CO + a)
        ni, no = tl.where(ci > CAP, IN, ci), tl.where(co > CAP, OUT, co)
        gi0, gi1 = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        go0, go1 = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        dot = tl.full((), 0.0, tl.float32)
        for ib in range(0, ni, T):
            ii, vi, fi, di0, di1 = _block(
                SI, QI, JI, PI, II, NI, CI, MI, a, ib, IN, CAP, T, FLOOR_I
            )
            ri = tl.full((T,), 0.0, tl.float32)
            for ob in range(0, no, T):
                io, vo, fo, do0, do1 = _block(
                    SO, QO, JO, PO, IO, NO, CO, MO, a, ob, OUT, CAP, T, FLOOR_O
                )
                dw = tl.load(
                    DW + io[:, None] * IN + ii[None, :], vo[:, None] & vi[None, :], 0
                )
                ri += tl.sum(dw * fo[:, None], 0)
                ro = tl.sum(dw * fi[None, :], 1)
                go0 += tl.sum(ro * do0, 0)
                go1 += tl.sum(ro * do1, 0)
            dot += tl.sum(ri * fi, 0)
            gi0 += tl.sum(ri * di0, 0)
            gi1 += tl.sum(ri * di1, 0)
        amp = tl.load(Amp + a)
        tl.store(DP + a * 6, dot * tl.load(Damp + a * 2))
        tl.store(DP + a * 6 + 1, dot * tl.load(Damp + a * 2 + 1))
        tl.store(DP + a * 6 + 2, amp * gi0)
        tl.store(DP + a * 6 + 3, amp * gi1)
        tl.store(DP + a * 6 + 4, amp * go0)
        tl.store(DP + a * 6 + 5, amp * go1)
