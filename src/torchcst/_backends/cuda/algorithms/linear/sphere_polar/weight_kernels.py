"""Support-patch weight assembly and unsplit dW parameter VJP."""

import triton
import triton.language as tl

from .support_kernels import _block, _center_store


@triton.jit
def _weight(
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
    a,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    amp = tl.load(AMP + a)
    for ib in range(0, IN if FULL_I else CAP, T):
        if FULL_I or ib < tl.load(CI + a):
            ii, mi, vi, _, _, _, _, _ = _block(
                SI, QI, PI, II, FI, NI, CI, a, ib, IN, CAP, T, FULL_I, FLOOR_I
            )
            for ob in range(0, OUT if FULL_O else CAP, T):
                if FULL_O or ob < tl.load(CO + a):
                    oo, mo, uo, _, _, _, _, _ = _block(
                        SO, QO, PO, IO, FO, NO, CO, a, ob, OUT, CAP, T, FULL_O, FLOOR_O
                    )
                    mask = (
                        mo[:, None]
                        & mi[None, :]
                        & (uo[:, None] != 0)
                        & (vi[None, :] != 0)
                    )
                    tl.atomic_add(
                        W + oo[:, None] * IN + ii[None, :],
                        amp * uo[:, None] * vi[None, :],
                        mask,
                        sem="relaxed",
                    )


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
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(CI + a) > CAP:
        if tl.load(CO + a) > CAP:
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
                a,
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
                a,
                IN,
                OUT,
                CAP,
                T,
                True,
                False,
                FLOOR_I,
                FLOOR_O,
            )
    else:
        if tl.load(CO + a) > CAP:
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
                a,
                IN,
                OUT,
                CAP,
                T,
                False,
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
                a,
                IN,
                OUT,
                CAP,
                T,
                False,
                False,
                FLOOR_I,
                FLOOR_O,
            )


@triton.jit
def _vjp(
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
    a,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    gi0, gi1, gi2, go0, go1, go2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ni0, ni1, ni2, no0, no1, no2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dot = 0.0
    normi = tl.load(NI + a)
    normo = tl.load(NO + a)
    for ib in range(0, IN if FULL_I else CAP, T):
        if FULL_I or ib < tl.load(CI + a):
            ii, mi, vi, gap, di0, di1, di2, precision = _block(
                SI, QI, PI, II, FI, NI, CI, a, ib, IN, CAP, T, FULL_I, FLOOR_I
            )
            coeff = 6.0 * precision * gap * gap / tl.maximum(normi, FLOOR_I)
            di0, di1, di2 = coeff * di0, coeff * di1, coeff * di2
            ni0 += tl.sum(vi * di0, 0)
            ni1 += tl.sum(vi * di1, 0)
            ni2 += tl.sum(vi * di2, 0)
            ri = tl.full((T,), 0, tl.float32)
            for ob in range(0, OUT if FULL_O else CAP, T):
                if FULL_O or ob < tl.load(CO + a):
                    oo, mo, uo, gapo, do0, do1, do2, preco = _block(
                        SO, QO, PO, IO, FO, NO, CO, a, ob, OUT, CAP, T, FULL_O, FLOOR_O
                    )
                    coeffo = 6.0 * preco * gapo * gapo / tl.maximum(normo, FLOOR_O)
                    do0, do1, do2 = coeffo * do0, coeffo * do1, coeffo * do2
                    dw = tl.load(
                        DW + oo[:, None] * IN + ii[None, :],
                        mo[:, None] & mi[None, :],
                        0,
                    )
                    ri += tl.sum(dw * uo[:, None], 0)
                    ro = tl.sum(dw * vi[None, :], 1)
                    go0 += tl.sum(ro * do0, 0)
                    go1 += tl.sum(ro * do1, 0)
                    go2 += tl.sum(ro * do2, 0)
                    if ib == 0:
                        no0 += tl.sum(uo * do0, 0)
                        no1 += tl.sum(uo * do1, 0)
                        no2 += tl.sum(uo * do2, 0)
            dot += tl.sum(ri * vi, 0)
            gi0 += tl.sum(ri * di0, 0)
            gi1 += tl.sum(ri * di1, 0)
            gi2 += tl.sum(ri * di2, 0)
    amp = tl.load(AMP + a)
    projectioni = tl.where(normi >= FLOOR_I, dot, 0.0)
    projectiono = tl.where(normo >= FLOOR_O, dot, 0.0)
    _center_store(
        JI,
        DP,
        a,
        amp * (gi0 - projectioni * ni0),
        amp * (gi1 - projectioni * ni1),
        amp * (gi2 - projectioni * ni2),
        2,
    )
    _center_store(
        JO,
        DP,
        a,
        amp * (go0 - projectiono * no0),
        amp * (go1 - projectiono * no1),
        amp * (go2 - projectiono * no2),
        4,
    )
    tl.store(DP + a * 6, dot * tl.load(DAMP + a * 2))
    tl.store(DP + a * 6 + 1, dot * tl.load(DAMP + a * 2 + 1))


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
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
):
    a = tl.program_id(0)
    if tl.load(CI + a) > CAP:
        if tl.load(CO + a) > CAP:
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
                a,
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
                a,
                IN,
                OUT,
                CAP,
                T,
                True,
                False,
                FLOOR_I,
                FLOOR_O,
            )
    else:
        if tl.load(CO + a) > CAP:
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
                a,
                IN,
                OUT,
                CAP,
                T,
                False,
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
                a,
                IN,
                OUT,
                CAP,
                T,
                False,
                False,
                FLOOR_I,
                FLOOR_O,
            )
