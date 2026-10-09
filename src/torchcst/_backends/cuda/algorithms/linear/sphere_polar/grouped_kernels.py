"""Independent atom lanes; complete overflow keeps the existing scalar kernels."""

import triton
import triton.language as tl

from .weight_kernels import _vjp, _weight


@triton.jit
def _block(
    S,
    Q,
    P,
    Index,
    Phi,
    a,
    live,
    count,
    base,
    N: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
):
    off = base + tl.arange(0, T)
    valid = live[:, None] & (off[None, :] < count[:, None])
    idx = tl.load(Index + a[:, None] * CAP + off[None, :], valid, 0).to(tl.int32)
    d0 = tl.load(S + idx * 3, valid, 0) - tl.load(Q + a * 3, live, 0)[:, None]
    d1 = tl.load(S + idx * 3 + 1, valid, 0) - tl.load(Q + a * 3 + 1, live, 0)[:, None]
    d2 = tl.load(S + idx * 3 + 2, valid, 0) - tl.load(Q + a * 3 + 2, live, 0)[:, None]
    precision = tl.load(P + a, live, 0)
    gap = tl.maximum(1.0 - ((d0 * d0 + d1 * d1) + d2 * d2) * precision[:, None], 0.0)
    gap = tl.where(valid, gap, 0.0)
    phi = tl.load(Phi + a[:, None] * CAP + off[None, :], valid, 0)
    return idx, valid, phi, gap, d0, d1, d2, precision


@triton.jit
def _center_store(J, DP, a, live, b0, b1, b2, COL: tl.constexpr):
    g0 = (
        b0 * tl.load(J + a * 6, live, 0)
        + b1 * tl.load(J + a * 6 + 2, live, 0)
        + b2 * tl.load(J + a * 6 + 4, live, 0)
    )
    g1 = (
        b0 * tl.load(J + a * 6 + 1, live, 0)
        + b1 * tl.load(J + a * 6 + 3, live, 0)
        + b2 * tl.load(J + a * 6 + 5, live, 0)
    )
    tl.store(DP + a * 6 + COL, g0, live)
    tl.store(DP + a * 6 + COL + 1, g1, live)


@triton.jit
def assemble(
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
    W,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    ci = tl.load(CI + a, real, 0)
    co = tl.load(CO + a, real, 0)
    live = real & (ci <= CAP) & (co <= CAP)
    amp = tl.load(AMP + a, live, 0)
    ni = tl.where(live, ci, 0)
    no = tl.where(live, co, 0)
    for ib in range(0, tl.max(ni, 0), T):
        ii, mi, vi, _, _, _, _, _ = _block(
            SI, QI, PI, II, FI, a, live, ni, ib, IN, CAP, T
        )
        for ob in range(0, tl.max(no, 0), T):
            oo, mo, uo, _, _, _, _, _ = _block(
                SO, QO, PO, IO, FO, a, live, no, ob, OUT, CAP, T
            )
            mask = (
                mo[:, :, None]
                & mi[:, None, :]
                & (uo[:, :, None] != 0)
                & (vi[:, None, :] != 0)
            )
            tl.atomic_add(
                W + oo[:, :, None] * IN + ii[:, None, :],
                amp[:, None, None] * uo[:, :, None] * vi[:, None, :],
                mask,
                sem="relaxed",
            )
    # Scalar overflow is exact and does not expand non-overflow atom work.
    # Every branch predicate is CTA-uniform, including incomplete group tails.
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A:
            fulli = tl.load(CI + atom) > CAP
            fullo = tl.load(CO + atom) > CAP
            if fulli:
                if fullo:
                    _weight(
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
                        W,
                        atom,
                        IN,
                        OUT,
                        CAP,
                        T,
                        True,
                        True,
                        FLOOR_I,
                        FLOOR_O,
                    )
                else:
                    _weight(
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
                        W,
                        atom,
                        IN,
                        OUT,
                        CAP,
                        T,
                        True,
                        False,
                        FLOOR_I,
                        FLOOR_O,
                    )
            elif fullo:
                _weight(
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
                    W,
                    atom,
                    IN,
                    OUT,
                    CAP,
                    T,
                    False,
                    True,
                    FLOOR_I,
                    FLOOR_O,
                )


@triton.jit
def vjp(
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
    DW,
    DP,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    ci = tl.load(CI + a, real, 0)
    co = tl.load(CO + a, real, 0)
    live = real & (ci <= CAP) & (co <= CAP)
    ni = tl.where(live, ci, 0)
    no = tl.where(live, co, 0)
    normi = tl.load(NI + a, live, 0)
    normo = tl.load(NO + a, live, 0)
    gi0, gi1, gi2 = (
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
    )
    go0, go1, go2 = (
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
    )
    ni0, ni1, ni2 = (
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
    )
    no0, no1, no2 = (
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
        tl.full((G,), 0.0, tl.float32),
    )
    dot = tl.full((G,), 0.0, tl.float32)
    for ib in range(0, tl.max(ni, 0), T):
        ii, mi, vi, gap, di0, di1, di2, precision = _block(
            SI, QI, PI, II, FI, a, live, ni, ib, IN, CAP, T
        )
        coeff = (
            6.0 * precision[:, None] * gap * gap / tl.maximum(normi[:, None], FLOOR_I)
        )
        di0, di1, di2 = coeff * di0, coeff * di1, coeff * di2
        ni0 += tl.sum(vi * di0, 1)
        ni1 += tl.sum(vi * di1, 1)
        ni2 += tl.sum(vi * di2, 1)
        ri = tl.full((G, T), 0.0, tl.float32)
        for ob in range(0, tl.max(no, 0), T):
            oo, mo, uo, gapo, do0, do1, do2, preco = _block(
                SO, QO, PO, IO, FO, a, live, no, ob, OUT, CAP, T
            )
            coeffo = (
                6.0 * preco[:, None] * gapo * gapo / tl.maximum(normo[:, None], FLOOR_O)
            )
            do0, do1, do2 = coeffo * do0, coeffo * do1, coeffo * do2
            dw = tl.load(
                DW + oo[:, :, None] * IN + ii[:, None, :],
                mo[:, :, None] & mi[:, None, :],
                0,
            )
            ri += tl.sum(dw * uo[:, :, None], 1)
            ro = tl.sum(dw * vi[:, None, :], 2)
            go0 += tl.sum(ro * do0, 1)
            go1 += tl.sum(ro * do1, 1)
            go2 += tl.sum(ro * do2, 1)
            if ib == 0:
                no0 += tl.sum(uo * do0, 1)
                no1 += tl.sum(uo * do1, 1)
                no2 += tl.sum(uo * do2, 1)
        dot += tl.sum(ri * vi, 1)
        gi0 += tl.sum(ri * di0, 1)
        gi1 += tl.sum(ri * di1, 1)
        gi2 += tl.sum(ri * di2, 1)
    amp = tl.load(AMP + a, live, 0)
    projectioni = tl.where(normi >= FLOOR_I, dot, 0.0)
    projectiono = tl.where(normo >= FLOOR_O, dot, 0.0)
    _center_store(
        JI,
        DP,
        a,
        live,
        amp * (gi0 - projectioni * ni0),
        amp * (gi1 - projectioni * ni1),
        amp * (gi2 - projectioni * ni2),
        2,
    )
    _center_store(
        JO,
        DP,
        a,
        live,
        amp * (go0 - projectiono * no0),
        amp * (go1 - projectiono * no1),
        amp * (go2 - projectiono * no2),
        4,
    )
    tl.store(DP + a * 6, dot * tl.load(DAMP + a * 2, live, 0), live)
    tl.store(DP + a * 6 + 1, dot * tl.load(DAMP + a * 2 + 1, live, 0), live)
    # Scalar overflow is exact and does not expand non-overflow atom work.
    # Every branch predicate is CTA-uniform, including incomplete group tails.
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A:
            fulli = tl.load(CI + atom) > CAP
            fullo = tl.load(CO + atom) > CAP
            if fulli:
                if fullo:
                    _vjp(
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
                        DW,
                        DP,
                        atom,
                        IN,
                        OUT,
                        CAP,
                        T,
                        True,
                        True,
                        FLOOR_I,
                        FLOOR_O,
                    )
                else:
                    _vjp(
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
                        DW,
                        DP,
                        atom,
                        IN,
                        OUT,
                        CAP,
                        T,
                        True,
                        False,
                        FLOOR_I,
                        FLOOR_O,
                    )
            elif fullo:
                _vjp(
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
                    DW,
                    DP,
                    atom,
                    IN,
                    OUT,
                    CAP,
                    T,
                    False,
                    True,
                    FLOOR_I,
                    FLOOR_O,
                )
