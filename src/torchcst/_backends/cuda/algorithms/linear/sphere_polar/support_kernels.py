"""Complete support packing, with exact full traversal on capacity overflow."""

import triton
import triton.language as tl

from .kernels import _raw


@triton.jit
def pack(
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    Count,
    N: tl.constexpr,
    CAP: tl.constexpr,
    V: tl.constexpr,
    FLOOR: tl.constexpr,
):
    a = tl.program_id(0)
    site = tl.arange(0, V)
    gap, _, _, _, _ = _raw(S, Q, P, a, site, N)
    raw = gap * gap * gap
    norm = tl.sqrt(tl.sum(raw * raw, 0))
    present = gap > 0
    rank = tl.cumsum(present.to(tl.int32), 0) - 1
    count = tl.sum(present.to(tl.int32), 0)
    # Overflow never drops an atom/site; contractions take the full path.
    mask = present & (count <= CAP)
    tl.store(Index + a * CAP + rank, site, mask)
    tl.store(Phi + a * CAP + rank, raw / tl.maximum(norm, FLOOR), mask)
    tl.store(Norm + a, norm)
    tl.store(Count + a, count)


@triton.jit
def _block(
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    Count,
    a,
    base,
    N: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FULL: tl.constexpr,
    FLOOR: tl.constexpr,
):
    off = base + tl.arange(0, T)
    if FULL:
        idx = off
        valid = off < N
    else:
        valid = off < tl.load(Count + a)
        idx = tl.load(Index + a * CAP + off, valid, 0)
    gap, d0, d1, d2, precision = _raw(S, Q, P, a, idx, N)
    gap = tl.where(valid, gap, 0.0)
    if FULL:
        phi = gap * gap * gap / tl.maximum(tl.load(Norm + a), FLOOR)
    else:
        phi = tl.load(Phi + a * CAP + off, valid, 0)
    return idx, valid, phi, gap, d0, d1, d2, precision


@triton.jit
def _one_forward(
    X,
    SI,
    QI,
    PI,
    II,
    FI,
    NI,
    CI,
    SO,
    QO,
    PO,
    IO,
    FO,
    NO,
    CO,
    AMP,
    H,
    Y,
    a,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
):
    batch = tl.arange(0, PB)
    h = tl.full((PB,), 0, tl.float32)
    for base in range(0, IN if FULL_I else CAP, T):
        if FULL_I or base < tl.load(CI + a):
            idx, valid, phi, _, _, _, _, _ = _block(
                SI, QI, PI, II, FI, NI, CI, a, base, IN, CAP, T, FULL_I, FLOOR_I
            )
            xx = tl.load(
                X + batch[:, None] * IN + idx[None, :],
                (batch[:, None] < B) & valid[None, :],
                0,
            )
            h += tl.sum(xx * phi[None, :], 1)
    if SAVE_H:
        tl.store(H + a * B + batch, h, batch < B)
    h *= tl.load(AMP + a)
    for base in range(0, OUT if FULL_O else CAP, T):
        if FULL_O or base < tl.load(CO + a):
            idx, valid, phi, _, _, _, _, _ = _block(
                SO, QO, PO, IO, FO, NO, CO, a, base, OUT, CAP, T, FULL_O, FLOOR_O
            )
            mask = (batch[:, None] < B) & valid[None, :] & (phi[None, :] != 0)
            tl.atomic_add(
                Y + batch[:, None] * OUT + idx[None, :],
                h[:, None] * phi[None, :],
                mask,
                sem="relaxed",
            )


@triton.jit
def forward(
    X,
    SI,
    QI,
    PI,
    II,
    FI,
    NI,
    CI,
    SO,
    QO,
    PO,
    IO,
    FO,
    NO,
    CO,
    AMP,
    H,
    Y,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(CI + a) > CAP:
        if tl.load(CO + a) > CAP:
            _one_forward(
                X,
                SI,
                QI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                H,
                Y,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                True,
                True,
                FLOOR_I,
                FLOOR_O,
                SAVE_H,
            )
        else:
            _one_forward(
                X,
                SI,
                QI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                H,
                Y,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                True,
                False,
                FLOOR_I,
                FLOOR_O,
                SAVE_H,
            )
    else:
        if tl.load(CO + a) > CAP:
            _one_forward(
                X,
                SI,
                QI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                H,
                Y,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                False,
                True,
                FLOOR_I,
                FLOOR_O,
                SAVE_H,
            )
        else:
            _one_forward(
                X,
                SI,
                QI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                H,
                Y,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                False,
                False,
                FLOOR_I,
                FLOOR_O,
                SAVE_H,
            )


@triton.jit
def _center_store(J, DP, a, b0, b1, b2, COL: tl.constexpr):
    g0 = (
        b0 * tl.load(J + a * 6)
        + b1 * tl.load(J + a * 6 + 2)
        + b2 * tl.load(J + a * 6 + 4)
    )
    g1 = (
        b0 * tl.load(J + a * 6 + 1)
        + b1 * tl.load(J + a * 6 + 3)
        + b2 * tl.load(J + a * 6 + 5)
    )
    tl.store(DP + a * 6 + COL, g0)
    tl.store(DP + a * 6 + COL + 1, g1)


@triton.jit
def _one_backward(
    X,
    DY,
    SI,
    QI,
    JI,
    PI,
    II,
    FI,
    NI,
    CI,
    SO,
    QO,
    JO,
    PO,
    IO,
    FO,
    NO,
    CO,
    AMP,
    DAMP,
    H,
    DX,
    DP,
    a,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
):
    batch = tl.arange(0, PB)
    g = tl.full((PB,), 0, tl.float32)
    for base in range(0, OUT if FULL_O else CAP, T):
        if FULL_O or base < tl.load(CO + a):
            idx, valid, phi, _, _, _, _, _ = _block(
                SO, QO, PO, IO, FO, NO, CO, a, base, OUT, CAP, T, FULL_O, FLOOR_O
            )
            yy = tl.load(
                DY + batch[:, None] * OUT + idx[None, :],
                (batch[:, None] < B) & valid[None, :],
                0,
            )
            g += tl.sum(yy * phi[None, :], 1)
    amp = tl.load(AMP + a)
    if NEED_P:
        h = tl.load(H + a * B + batch, batch < B, 0)
        dot = tl.sum(h * g, 0)
        projection = amp * dot
        tl.store(DP + a * 6, dot * tl.load(DAMP + a * 2))
        tl.store(DP + a * 6 + 1, dot * tl.load(DAMP + a * 2 + 1))
    b0, b1, b2 = 0.0, 0.0, 0.0
    for base in range(0, IN if FULL_I else CAP, T):
        if FULL_I or base < tl.load(CI + a):
            idx, valid, phi, gap, d0, d1, d2, precision = _block(
                SI, QI, PI, II, FI, NI, CI, a, base, IN, CAP, T, FULL_I, FLOOR_I
            )
            mask = (batch[:, None] < B) & valid[None, :]
            if NEED_X:
                tl.atomic_add(
                    DX + batch[:, None] * IN + idx[None, :],
                    (amp * g)[:, None] * phi[None, :],
                    mask & (phi[None, :] != 0),
                    sem="relaxed",
                )
            if NEED_P:
                xx = tl.load(X + batch[:, None] * IN + idx[None, :], mask, 0)
                weight = amp * tl.sum(xx * g[:, None], 0)
                norm = tl.load(NI + a)
                weight -= tl.where(norm >= FLOOR_I, phi * projection, 0.0)
                coeff = 6.0 * precision * gap * gap * weight / tl.maximum(norm, FLOOR_I)
                b0 += tl.sum(coeff * d0, 0)
                b1 += tl.sum(coeff * d1, 0)
                b2 += tl.sum(coeff * d2, 0)
    if NEED_P:
        _center_store(JI, DP, a, b0, b1, b2, 2)
        b0, b1, b2 = 0.0, 0.0, 0.0
        for base in range(0, OUT if FULL_O else CAP, T):
            if FULL_O or base < tl.load(CO + a):
                idx, valid, phi, gap, d0, d1, d2, precision = _block(
                    SO, QO, PO, IO, FO, NO, CO, a, base, OUT, CAP, T, FULL_O, FLOOR_O
                )
                yy = tl.load(
                    DY + batch[:, None] * OUT + idx[None, :],
                    (batch[:, None] < B) & valid[None, :],
                    0,
                )
                weight = amp * tl.sum(yy * h[:, None], 0)
                norm = tl.load(NO + a)
                weight -= tl.where(norm >= FLOOR_O, phi * projection, 0.0)
                coeff = 6.0 * precision * gap * gap * weight / tl.maximum(norm, FLOOR_O)
                b0 += tl.sum(coeff * d0, 0)
                b1 += tl.sum(coeff * d1, 0)
                b2 += tl.sum(coeff * d2, 0)
        _center_store(JO, DP, a, b0, b1, b2, 4)


@triton.jit
def backward(
    X,
    DY,
    SI,
    QI,
    JI,
    PI,
    II,
    FI,
    NI,
    CI,
    SO,
    QO,
    JO,
    PO,
    IO,
    FO,
    NO,
    CO,
    AMP,
    DAMP,
    H,
    DX,
    DP,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(CI + a) > CAP:
        if tl.load(CO + a) > CAP:
            _one_backward(
                X,
                DY,
                SI,
                QI,
                JI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                JO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                DAMP,
                H,
                DX,
                DP,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                True,
                True,
                FLOOR_I,
                FLOOR_O,
                NEED_X,
                NEED_P,
            )
        else:
            _one_backward(
                X,
                DY,
                SI,
                QI,
                JI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                JO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                DAMP,
                H,
                DX,
                DP,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                True,
                False,
                FLOOR_I,
                FLOOR_O,
                NEED_X,
                NEED_P,
            )
    else:
        if tl.load(CO + a) > CAP:
            _one_backward(
                X,
                DY,
                SI,
                QI,
                JI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                JO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                DAMP,
                H,
                DX,
                DP,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                False,
                True,
                FLOOR_I,
                FLOOR_O,
                NEED_X,
                NEED_P,
            )
        else:
            _one_backward(
                X,
                DY,
                SI,
                QI,
                JI,
                PI,
                II,
                FI,
                NI,
                CI,
                SO,
                QO,
                JO,
                PO,
                IO,
                FO,
                NO,
                CO,
                AMP,
                DAMP,
                H,
                DX,
                DP,
                a,
                B,
                IN,
                OUT,
                CAP,
                PB,
                T,
                False,
                False,
                FLOOR_I,
                FLOOR_O,
                NEED_X,
                NEED_P,
            )
