"""Atom-owned fused contractions. H/G are register values, Y/dX use atomics.

No spatial routing is assumed: GROUP contains consecutive original atoms.
These kernels isolate lifetime/fusion effects before support-overlap grouping.
"""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent
from ..periodic_product.kernels import _index, _periodic_raw


@tr.jit
def _compact_wrap_safe(
    center, inv, valid, O: tl.constexpr, L: tl.constexpr, N: tl.constexpr
):
    if O == 0.0 and L == float(N) and N >= 1 and N <= 8192:
        # Unit-spacing sites are exact integers. Keep support away from the
        # half-period rounding ambiguity and all raw intermediates finite.
        safe = (
            (center >= 0.0) & (center < L) & (inv >= 16.0 / (L * L)) & (inv <= 2.0**100)
        )
        return tl.min((safe | ~valid).to(tl.int32), 0) != 0
    else:
        return tl.full((), False, tl.int1)


@tr.jit
def _compact_periodic_raw(
    j, center, inv, safe, O: tl.constexpr, L: tl.constexpr, N: tl.constexpr
):
    if safe:
        delta = O + _index(j, N) * (L / N) - center
        delta = tl.where(
            delta >= L / 2.0, delta - L, tl.where(delta < -L / 2.0, delta + L, delta)
        )
        return _raw(delta, inv)
    else:
        return _periodic_raw(j, center, inv, O, L, N)


@tr.jit
def _contract(
    X,
    P,
    a,
    rows,
    valid,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    OUTPUT: tl.constexpr,
    DERIVATIVE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    RECIPROCAL: tl.constexpr = False,
    COMPACT_WRAP: tl.constexpr = False,
):
    k = tl.arange(0, BK)
    inv = tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    high = tl.where(other_high > other_low, high, low)
    if COMPACT_WRAP:
        wrap_safe = _compact_wrap_safe(center, inv, valid, O, L, N)
    if DERIVATIVE:
        loggrad = tl.load(P + (7 if OUTPUT else 6) * A + a, valid, 0)
        flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    elif RECIPROCAL:
        flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    if RECIPROCAL:
        # Triweight values lie in [0, 1], N <= 8192. Individual norms
        # can lie below the joint floor: use division for these and singleton
        # groups to retain the existing floor and one-point evaluation.
        # A normal reciprocal is safe in that domain; preserve exact division
        # for an unexpected packed norm. Do not move normalization past sums.
        reciprocal_safe = tl.min(
            (
                (norm >= 1e-6) & (norm <= 128.0) & ((flags & (2 if OUTPUT else 1)) == 0)
            ).to(tl.int32),
            0,
        )
        inverse_norm = tl.div_rn(1.0, norm)
    acc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    grad = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(high - low, 0), 0), BK):
        j = low[:, None] + start + k[None, :]
        if COMPACT_WRAP:
            v, dv = _compact_periodic_raw(
                j, center[:, None], inv[:, None], wrap_safe, O, L, N
            )
        else:
            v, dv = _periodic_raw(j, center[:, None], inv[:, None], O, L, N)
        if RECIPROCAL:
            if reciprocal_safe:
                v = v * inverse_norm[:, None]
                if DERIVATIVE:
                    dv = dv * inverse_norm[:, None]
            else:
                v = tl.div_rn(v, norm[:, None])
                if DERIVATIVE:
                    dv = tl.div_rn(dv, norm[:, None])
        else:
            v = tl.div_rn(v, norm[:, None])
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < high[:, None, None])
        )
        xx = tl.load(
            X + rows[None, :, None] * X0 + _index(j, N)[:, None, :] * X1, mask, 0.0
        )
        acc += xx * v[:, None, :]
        if DERIVATIVE:
            if not RECIPROCAL:
                dv = tl.div_rn(dv, norm[:, None])
            dv = tl.where(
                (flags[:, None] & (2 if OUTPUT else 1)) != 0,
                0.0,
                dv - loggrad[:, None] * v,
            )
            grad += xx * dv[:, None, :]
    return tl.sum(acc, 2), tl.sum(grad, 2)


@tr.jit
def _scatter(
    T,
    P,
    Y,
    a,
    rows,
    valid,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    OUTPUT: tl.constexpr,
    BK: tl.constexpr,
):
    k = tl.arange(0, BK)
    amp, inv = tl.load(P + a, valid, 0), tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    for start in range(0, tl.max(tl.maximum(high - low, 0), 0), BK):
        j = low[:, None] + start + k[None, :]
        v, _ = _periodic_raw(j, center[:, None], inv[:, None], O, L, N)
        v = tl.div_rn(v, norm[:, None]) * amp[:, None]
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < high[:, None, None])
        )
        tl.atomic_add(
            Y + rows[None, :, None] * N + _index(j, N)[:, None, :],
            T[:, :, None] * v[:, None, :],
            mask,
            sem="relaxed",
        )


@tr.jit
def forward(
    X,
    P,
    Y,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    valid = a < A
    h, _ = _contract(
        X, P, a, rows, valid, A, B, NI, LI, OI, X0, X1, False, False, BM, BK, GROUP
    )
    _scatter(h, P, Y, a, rows, valid, A, B, NO, LO, OO, True, BK)


@tr.jit
def backward(
    X,
    DY,
    P,
    DX,
    Partial,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    tile = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    valid = a < A
    g, dg = _contract(
        DY, P, a, rows, valid, A, B, NO, LO, OO, D0, D1, True, NEED_P, BM, BK, GROUP
    )
    if NEED_X:
        _scatter(g, P, DX, a, rows, valid, A, B, NI, LI, OI, False, BK)
    if NEED_P:
        h, dh = _contract(
            X, P, a, rows, valid, A, B, NI, LI, OI, X0, X1, False, True, BM, BK, GROUP
        )
        amp = tl.load(P + a, valid, 0)
        tl.store(Partial + tile * 3 * A + a, tl.sum(h * g, 1), valid)
        tl.store(Partial + tile * 3 * A + A + a, amp * tl.sum(g * dh, 1), valid)
        tl.store(Partial + tile * 3 * A + 2 * A + a, amp * tl.sum(h * dg, 1), valid)


@tr.jit
def reduce_parameters(
    Partial,
    DP,
    Source,
    AmplitudeMax,
    A: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    t = tl.arange(0, BT)
    mask = (a[:, None] < A) & (t[None, :] < TILES)
    base = Partial + t[None, :] * 3 * A + a[:, None]
    da = tl.sum(tl.load(base, mask, 0), 1)
    dci = tl.sum(tl.load(base + A, mask, 0), 1)
    dco = tl.sum(tl.load(base + 2 * A, mask, 0), 1)
    _store_param_cotangent(
        DP, Source, AmplitudeMax, a, a < A, da, dci, dco, True, CENTER_OUTPUT_FIRST=True
    )
