"""Independent atom lanes and batch reductions, without U/V, W/dW or tl.dot."""

import triton
import triton.language as tl

from .grouped_kernels import _center_store


@triton.jit
def _block(
    S,
    Q,
    P,
    Index,
    Phi,
    Norm,
    a,
    live,
    count,
    base,
    N: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FULL: tl.constexpr,
    FLOOR: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    off = base + tl.arange(0, T)
    if FULL:
        idx = tl.broadcast_to(off[None, :], (a.shape[0], T))
        valid = live[:, None] & (off[None, :] < N)
    else:
        valid = live[:, None] & (off[None, :] < count[:, None])
        idx = tl.load(Index + a[:, None] * CAP + off[None, :], valid, 0).to(tl.int32)
    d0 = tl.load(S + idx * 3, valid, 0) - tl.load(Q + a * 3, live, 0)[:, None]
    d1 = tl.load(S + idx * 3 + 1, valid, 0) - tl.load(Q + a * 3 + 1, live, 0)[:, None]
    d2 = tl.load(S + idx * 3 + 2, valid, 0) - tl.load(Q + a * 3 + 2, live, 0)[:, None]
    precision = tl.load(P + a, live, 0)
    gap = tl.maximum(1.0 - ((d0 * d0 + d1 * d1) + d2 * d2) * precision[:, None], 0.0)
    gap = tl.where(valid, gap, 0.0)
    if FULL or RECOMPUTE:
        # Saved geometry/norm make recomputation independent of live state.
        phi = gap * gap * gap / tl.maximum(tl.load(Norm + a, live, 0)[:, None], FLOOR)
    else:
        phi = tl.load(Phi + a[:, None] * CAP + off[None, :], valid, 0)
    return idx, valid, phi, gap, d0, d1, d2, precision


@triton.jit
def _forward(
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
    live,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    batch = tl.arange(0, PB)
    ci = tl.load(CI + a, live, 0)
    co = tl.load(CO + a, live, 0)
    h = tl.full((G, PB), 0.0, tl.float32)
    for base in range(0, IN if FULL_I else tl.max(ci, 0), T):
        idx, valid, phi, _, _, _, _, _ = _block(
            SI,
            QI,
            PI,
            II,
            FI,
            NI,
            a,
            live,
            ci,
            base,
            IN,
            CAP,
            T,
            FULL_I,
            FLOOR_I,
            RECOMPUTE,
        )
        xx = tl.load(
            X + batch[None, :, None] * IN + idx[:, None, :],
            (batch[None, :, None] < B) & valid[:, None, :],
            0,
        )
        h += tl.sum(xx * phi[:, None, :], 2)
    if SAVE_H:
        tl.store(
            H + a[:, None] * B + batch[None, :], h, live[:, None] & (batch[None, :] < B)
        )
    h *= tl.load(AMP + a, live, 0)[:, None]
    for base in range(0, OUT if FULL_O else tl.max(co, 0), T):
        idx, valid, phi, _, _, _, _, _ = _block(
            SO,
            QO,
            PO,
            IO,
            FO,
            NO,
            a,
            live,
            co,
            base,
            OUT,
            CAP,
            T,
            FULL_O,
            FLOOR_O,
            RECOMPUTE,
        )
        tl.atomic_add(
            Y + batch[None, :, None] * OUT + idx[:, None, :],
            h[:, :, None] * phi[:, None, :],
            (batch[None, :, None] < B) & valid[:, None, :] & (phi[:, None, :] != 0),
            sem="relaxed",
        )


@triton.jit
def _backward(
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
    live,
    B: tl.constexpr,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    CAP: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    FULL_I: tl.constexpr,
    FULL_O: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
    MERGE: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    batch = tl.arange(0, PB)
    ci = tl.load(CI + a, live, 0)
    co = tl.load(CO + a, live, 0)
    amp = tl.load(AMP + a, live, 0)
    g = tl.full((G, PB), 0.0, tl.float32)
    if NEED_P:
        h = tl.load(
            H + a[:, None] * B + batch[None, :], live[:, None] & (batch[None, :] < B), 0
        )
        normo = tl.load(NO + a, live, 0)
    if NEED_P and MERGE:
        c0 = tl.full((G,), 0.0, tl.float32)
        c1 = tl.full((G,), 0.0, tl.float32)
        c2 = tl.full((G,), 0.0, tl.float32)
        n0 = tl.full((G,), 0.0, tl.float32)
        n1 = tl.full((G,), 0.0, tl.float32)
        n2 = tl.full((G,), 0.0, tl.float32)
    for base in range(0, OUT if FULL_O else tl.max(co, 0), T):
        idx, valid, phi, gap, d0, d1, d2, precision = _block(
            SO,
            QO,
            PO,
            IO,
            FO,
            NO,
            a,
            live,
            co,
            base,
            OUT,
            CAP,
            T,
            FULL_O,
            FLOOR_O,
            RECOMPUTE,
        )
        yy = tl.load(
            DY + batch[None, :, None] * OUT + idx[:, None, :],
            (batch[None, :, None] < B) & valid[:, None, :],
            0,
        )
        g += tl.sum(yy * phi[:, None, :], 2)
        if NEED_P and MERGE:
            # Accumulate raw and normalization moments before dot(H,G) is known.
            # Each atom retains independent scalar moments; no reduction over G.
            coeff = (
                6.0
                * precision[:, None]
                * gap
                * gap
                / tl.maximum(normo[:, None], FLOOR_O)
            )
            t0, t1, t2 = coeff * d0, coeff * d1, coeff * d2
            weight = amp[:, None] * tl.sum(yy * h[:, :, None], 1)
            c0 += tl.sum(weight * t0, 1)
            c1 += tl.sum(weight * t1, 1)
            c2 += tl.sum(weight * t2, 1)
            n0 += tl.sum(phi * t0, 1)
            n1 += tl.sum(phi * t1, 1)
            n2 += tl.sum(phi * t2, 1)
    if NEED_P:
        dot = tl.sum(h * g, 1)
        projection = amp * dot
        tl.store(DP + a * 6, dot * tl.load(DAMP + a * 2, live, 0), live)
        tl.store(DP + a * 6 + 1, dot * tl.load(DAMP + a * 2 + 1, live, 0), live)
        b0 = tl.full((G,), 0.0, tl.float32)
        b1 = tl.full((G,), 0.0, tl.float32)
        b2 = tl.full((G,), 0.0, tl.float32)
        normi = tl.load(NI + a, live, 0)
    for base in range(0, IN if FULL_I else tl.max(ci, 0), T):
        idx, valid, phi, gap, d0, d1, d2, precision = _block(
            SI,
            QI,
            PI,
            II,
            FI,
            NI,
            a,
            live,
            ci,
            base,
            IN,
            CAP,
            T,
            FULL_I,
            FLOOR_I,
            RECOMPUTE,
        )
        mask = (batch[None, :, None] < B) & valid[:, None, :]
        if NEED_X:
            tl.atomic_add(
                DX + batch[None, :, None] * IN + idx[:, None, :],
                (amp[:, None] * g)[:, :, None] * phi[:, None, :],
                mask & (phi[:, None, :] != 0),
                sem="relaxed",
            )
        if NEED_P:
            xx = tl.load(X + batch[None, :, None] * IN + idx[:, None, :], mask, 0)
            weight = amp[:, None] * tl.sum(xx * g[:, :, None], 1)
            weight -= tl.where(
                normi[:, None] >= FLOOR_I, phi * projection[:, None], 0.0
            )
            coeff = (
                6.0
                * precision[:, None]
                * gap
                * gap
                * weight
                / tl.maximum(normi[:, None], FLOOR_I)
            )
            b0 += tl.sum(coeff * d0, 1)
            b1 += tl.sum(coeff * d1, 1)
            b2 += tl.sum(coeff * d2, 1)
    if NEED_P:
        _center_store(JI, DP, a, live, b0, b1, b2, 2)
        if MERGE:
            projected = tl.where(normo >= FLOOR_O, projection, 0.0)
            _center_store(
                JO,
                DP,
                a,
                live,
                c0 - projected * n0,
                c1 - projected * n1,
                c2 - projected * n2,
                4,
            )
        else:
            b0 = tl.full((G,), 0.0, tl.float32)
            b1 = tl.full((G,), 0.0, tl.float32)
            b2 = tl.full((G,), 0.0, tl.float32)
            for base in range(0, OUT if FULL_O else tl.max(co, 0), T):
                idx, valid, phi, gap, d0, d1, d2, precision = _block(
                    SO,
                    QO,
                    PO,
                    IO,
                    FO,
                    NO,
                    a,
                    live,
                    co,
                    base,
                    OUT,
                    CAP,
                    T,
                    FULL_O,
                    FLOOR_O,
                    RECOMPUTE,
                )
                yy = tl.load(
                    DY + batch[None, :, None] * OUT + idx[:, None, :],
                    (batch[None, :, None] < B) & valid[:, None, :],
                    0,
                )
                weight = amp[:, None] * tl.sum(yy * h[:, :, None], 1)
                weight -= tl.where(
                    normo[:, None] >= FLOOR_O, phi * projection[:, None], 0.0
                )
                coeff = (
                    6.0
                    * precision[:, None]
                    * gap
                    * gap
                    * weight
                    / tl.maximum(normo[:, None], FLOOR_O)
                )
                b0 += tl.sum(coeff * d0, 1)
                b1 += tl.sum(coeff * d1, 1)
                b2 += tl.sum(coeff * d2, 1)
            _center_store(JO, DP, a, live, b0, b1, b2, 4)


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
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    SAVE_H: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    ci = tl.load(CI + a, real, 0)
    co = tl.load(CO + a, real, 0)
    live = real & (ci <= CAP) & (co <= CAP)
    _forward(
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
        live,
        B,
        IN,
        OUT,
        CAP,
        PB,
        T,
        G,
        False,
        False,
        FLOOR_I,
        FLOOR_O,
        SAVE_H,
        RECOMPUTE,
    )
    # Overflow predicates are uniform and exclude the atom from packed work.
    # G1 retains the same single-pass ablation on full-axis fallback too.
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A:
            fulli = tl.load(CI + atom) > CAP
            fullo = tl.load(CO + atom) > CAP
            aa = atom + tl.arange(0, 1)
            active = aa < A
            if fulli:
                if fullo:
                    _forward(
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
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        True,
                        FLOOR_I,
                        FLOOR_O,
                        SAVE_H,
                        RECOMPUTE,
                    )
                else:
                    _forward(
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
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        False,
                        FLOOR_I,
                        FLOOR_O,
                        SAVE_H,
                        RECOMPUTE,
                    )
            elif fullo:
                _forward(
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
                    aa,
                    active,
                    B,
                    IN,
                    OUT,
                    CAP,
                    PB,
                    T,
                    1,
                    False,
                    True,
                    FLOOR_I,
                    FLOOR_O,
                    SAVE_H,
                    RECOMPUTE,
                )


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
    A: tl.constexpr,
    G: tl.constexpr,
    FLOOR_I: tl.constexpr,
    FLOOR_O: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
    MERGE: tl.constexpr,
    RECOMPUTE: tl.constexpr = False,
):
    a = tl.program_id(0) * G + tl.arange(0, G)
    real = a < A
    ci = tl.load(CI + a, real, 0)
    co = tl.load(CO + a, real, 0)
    live = real & (ci <= CAP) & (co <= CAP)
    _backward(
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
        live,
        B,
        IN,
        OUT,
        CAP,
        PB,
        T,
        G,
        False,
        False,
        FLOOR_I,
        FLOOR_O,
        NEED_X,
        NEED_P,
        MERGE,
        RECOMPUTE,
    )
    # Overflow predicates are uniform and exclude the atom from packed work.
    # G1 retains the same single-pass ablation on full-axis fallback too.
    for slot in range(G):
        atom = tl.program_id(0) * G + slot
        if atom < A:
            fulli = tl.load(CI + atom) > CAP
            fullo = tl.load(CO + atom) > CAP
            aa = atom + tl.arange(0, 1)
            active = aa < A
            if fulli:
                if fullo:
                    _backward(
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
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        True,
                        FLOOR_I,
                        FLOOR_O,
                        NEED_X,
                        NEED_P,
                        MERGE,
                        RECOMPUTE,
                    )
                else:
                    _backward(
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
                        aa,
                        active,
                        B,
                        IN,
                        OUT,
                        CAP,
                        PB,
                        T,
                        1,
                        True,
                        False,
                        FLOOR_I,
                        FLOOR_O,
                        NEED_X,
                        NEED_P,
                        MERGE,
                        RECOMPUTE,
                    )
            elif fullo:
                _backward(
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
                    aa,
                    active,
                    B,
                    IN,
                    OUT,
                    CAP,
                    PB,
                    T,
                    1,
                    False,
                    True,
                    FLOOR_I,
                    FLOOR_O,
                    NEED_X,
                    NEED_P,
                    MERGE,
                    RECOMPUTE,
                )
