"""Grouped exact normalization and per-atom H/G with atom-major scratch."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import (
    _candidate_span,
    _polar_atom,
    _raw,
    _store_param_cotangent,
)


@tr.jit
def _positions(j, Pitch, O: tl.constexpr, S: tl.constexpr, TILE: tl.constexpr):
    position = O + j * S
    if TILE:
        position = O + (j % TILE) * S + (j // TILE) * tl.load(Pitch)
    return position


@tr.jit
def _span(
    c, inv, Pitch, O: tl.constexpr, S: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr
):
    if not TILE:
        return _candidate_span(c, inv, O, S, N)
    else:
        pitch = tl.load(Pitch)
        radius = tl.sqrt_rn(tl.div_rn(1.0, inv))
        last = O + ((N - 1) % TILE) * S + ((N - 1) // TILE) * pitch
        error = (
            8.0
            * 1.1920928955078125e-7
            * (tl.abs(c) + tl.abs(O) + tl.abs(last) + radius)
        )
        precise = (
            (inv > 0.0) & (radius < float("inf")) & (error < S) & (pitch >= TILE * S)
        )
        low, high = c - O - radius, c - O + radius
        lt, ht = tl.floor(tl.div_rn(low, pitch)), tl.floor(tl.div_rn(high, pitch))
        lo = (
            lt * TILE
            + tl.minimum(tl.maximum(tl.floor(tl.div_rn(low - lt * pitch, S)), 0), TILE)
            - 2
        )
        hi = (
            ht * TILE
            + tl.minimum(tl.maximum(tl.ceil(tl.div_rn(high - ht * pitch, S)), 0), TILE)
            + 3
        )
        lo, hi = (
            tl.minimum(tl.maximum(lo, 0), N).to(tl.int32),
            tl.minimum(tl.maximum(hi, 0), N).to(tl.int32),
        )
        return tl.where(precise, lo, 0), tl.where(precise, tl.maximum(hi - lo, 0), N)


@tr.jit
def _stats(
    c,
    inv,
    valid,
    Pitch,
    O: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    GROUP: tl.constexpr,
    SUPPORT: tl.constexpr,
):
    if SUPPORT:
        lo, count = _span(c, inv, Pitch, O, S, N, TILE)
    else:
        lo, count = tl.full((GROUP,), 0, tl.int32), tl.full((GROUP,), N, tl.int32)
    count = tl.where(valid, count, 0)
    site = tl.arange(0, 32)
    vv, vd = (
        tl.full((GROUP, 32), 0.0, tl.float32),
        tl.full((GROUP, 32), 0.0, tl.float32),
    )
    first, last, live = (
        tl.full((GROUP,), N, tl.int32),
        tl.full((GROUP,), 0, tl.int32),
        tl.full((GROUP,), 0, tl.int32),
    )
    for start in range(0, tl.max(count, 0), 32):
        j = lo[:, None] + start + site[None, :]
        v, dv = _raw(_positions(j, Pitch, O, S, TILE) - c[:, None], inv[:, None])
        mask = valid[:, None] & (j < (lo + count)[:, None]) & (j < N)
        v, dv = tl.where(mask, v, 0.0), tl.where(mask, dv, 0.0)
        vv += v * v
        vd += v * dv
        first = tl.minimum(first, tl.min(tl.where(v > 0, j, N), 1))
        last = tl.maximum(last, tl.max(tl.where(v > 0, j + 1, 0), 1))
        live += tl.sum((v > 0).to(tl.int32), 1)
    return tl.sum(vv, 1), tl.sum(vd, 1), first, last, live


@tr.jit
def prepare(
    Source,
    P,
    A: tl.constexpr,
    KI: tl.constexpr,
    NO: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    PK: tl.constexpr,
    PN: tl.constexpr,
    Scalars,
    FLOOR: tl.constexpr,
    BOUNDS: tl.constexpr = True,
    STRIP_TILE: tl.constexpr = 0,
    Pitch=None,
    GROUP: tl.constexpr = 8,
    SUPPORT: tl.constexpr = True,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    amp, inv = _polar_atom(Source, tl.where(valid, a, 0), Scalars)
    co, ci = (
        tl.load(Source + 4 * a + 2, valid, 0),
        tl.load(Source + 4 * a + 3, valid, 0),
    )
    nv2, vd, ilo, ihi, iv = _stats(
        ci, inv, valid, Pitch, OI, S, KI, STRIP_TILE, GROUP, SUPPORT
    )
    nu2, ud, olo, ohi, ov = _stats(co, inv, valid, Pitch, OO, S, NO, 0, GROUP, SUPPORT)
    nv, nu = tl.sqrt(nv2), tl.sqrt(nu2)
    active = nv * nu >= FLOOR
    sv, su = tl.where(active, nv, tl.sqrt(FLOOR)), tl.where(active, nu, tl.sqrt(FLOOR))
    gv = tl.where(active, tl.div_rn(vd, tl.maximum(nv2, 1.1754943508222875e-38)), 0.0)
    gu = tl.where(active, tl.div_rn(ud, tl.maximum(nu2, 1.1754943508222875e-38)), 0.0)
    flags = (active & (iv == 1)).to(tl.int32) | ((active & (ov == 1)).to(tl.int32) << 1)
    tl.store(P + a, amp, valid)
    tl.store(P + A + a, inv, valid)
    tl.store(P + 2 * A + a, ci, valid)
    tl.store(P + 3 * A + a, co, valid)
    tl.store(P + 4 * A + a, sv, valid)
    tl.store(P + 5 * A + a, su, valid)
    tl.store(P + 6 * A + a, gv, valid)
    tl.store(P + 7 * A + a, gu, valid)
    tl.store(P + 8 * A + a, flags.to(tl.float32), valid)
    tl.store(P + 9 * A + a, ilo.to(tl.float32), valid)
    tl.store(P + 10 * A + a, ihi.to(tl.float32), valid)
    tl.store(P + 11 * A + a, olo.to(tl.float32), valid)
    tl.store(P + 12 * A + a, ohi.to(tl.float32), valid)


@tr.jit
def input_h(
    X,
    Views,
    Pitch,
    H,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    valid = a < A
    inv = tl.load(Views + A + a, valid, 1)
    ci, sv = tl.load(Views + 2 * A + a, valid, 0), tl.load(Views + 4 * A + a, valid, 1)
    lo, hi = (
        tl.load(Views + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(Views + 10 * A + a, valid, 0).to(tl.int32),
    )
    other_lo, other_hi = (
        tl.load(Views + 11 * A + a, valid, 0),
        tl.load(Views + 12 * A + a, valid, 0),
    )
    hi = tl.where(other_hi > other_lo, hi, lo)
    acc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(hi - lo, 0), 0), BK):
        j = lo[:, None] + start + k[None, :]
        v, _ = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
        v = tl.div_rn(v, sv[:, None])
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < hi[:, None, None])
            & (j[:, None, :] < N)
        )
        x = tl.load(X + rows[None, :, None] * X0 + j[:, None, :] * X1, mask, 0)
        acc += x * v[:, None, :]
    tl.store(
        H + a[:, None] * B + rows[None, :],
        tl.sum(acc, 2),
        valid[:, None] & (rows[None, :] < B),
    )


@tr.jit
def backward_atoms(
    X,
    DY,
    Views,
    Order,
    Inverse,
    H,
    G,
    DP,
    Source,
    AmplitudeMax,
    Pitch,
    A: tl.constexpr,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PARAMETERS: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    p = Views + 13 * A
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    amp, inv = tl.load(p + a, valid, 0), tl.load(p + A + a, valid, 1)
    ci, co = tl.load(p + 2 * A + a, valid, 0), tl.load(p + 3 * A + a, valid, 0)
    sv, su = tl.load(p + 4 * A + a, valid, 1), tl.load(p + 5 * A + a, valid, 1)
    gv, gu = tl.load(p + 6 * A + a, valid, 0), tl.load(p + 7 * A + a, valid, 0)
    flags = tl.load(p + 8 * A + a, valid, 0).to(tl.int32)
    ilo, ihi = (
        tl.load(p + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(p + 10 * A + a, valid, 0).to(tl.int32),
    )
    olo, ohi = (
        tl.load(p + 11 * A + a, valid, 0).to(tl.int32),
        tl.load(p + 12 * A + a, valid, 0).to(tl.int32),
    )
    ohi = tl.where(ihi > ilo, ohi, olo)
    acc, dc = (
        tl.full((GROUP, BM, BK), 0.0, tl.float32),
        tl.full((GROUP, BM, BK), 0.0, tl.float32),
    )
    for start in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), BK):
        i = olo[:, None] + start + k[None, :]
        u, du = _raw(OO + i * S - co[:, None], inv[:, None])
        u = tl.div_rn(u, su[:, None])
        du = tl.where(
            (flags[:, None] & 2) != 0, 0.0, tl.div_rn(du, su[:, None]) - gu[:, None] * u
        )
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (i[:, None, :] < ohi[:, None, None])
            & (i[:, None, :] < NO)
        )
        dy = tl.load(DY + rows[None, :, None] * D0 + i[:, None, :] * D1, mask, 0)
        acc += dy * u[:, None, :]
        if PARAMETERS:
            dc += dy * du[:, None, :]
    g = tl.sum(acc, 2)
    tl.store(
        G + a[:, None] * B + rows[None, :], g, valid[:, None] & (rows[None, :] < B)
    )
    if PARAMETERS:
        original = tl.load(Order + A + a, valid, 0)
        physical = tl.load(Inverse + original, valid, 0)
        h = tl.load(
            H + physical[:, None] * B + rows[None, :],
            valid[:, None] & (rows[None, :] < B),
            0,
        )
        da, dco = tl.sum(h * g, 1), amp * tl.sum(h * tl.sum(dc, 2), 1)
        hi = tl.where((ohi > olo) & ((flags & 1) == 0), ihi, ilo)
        hci = tl.full((GROUP, BM, BK), 0.0, tl.float32)
        for start in range(0, tl.max(tl.maximum(hi - ilo, 0), 0), BK):
            j = ilo[:, None] + start + k[None, :]
            v, dv = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
            dv = tl.div_rn(dv, sv[:, None]) - gv[:, None] * tl.div_rn(v, sv[:, None])
            mask = (
                valid[:, None, None]
                & (rows[None, :, None] < B)
                & (j[:, None, :] < hi[:, None, None])
                & (j[:, None, :] < NI)
            )
            x = tl.load(X + rows[None, :, None] * X0 + j[:, None, :] * X1, mask, 0)
            hci += x * dv[:, None, :]
        dci = amp * tl.sum(tl.sum(hci, 2) * g, 1)
        _store_param_cotangent(
            DP,
            Source,
            AmplitudeMax,
            original,
            valid,
            da,
            dci,
            dco,
            True,
            CENTER_OUTPUT_FIRST=True,
        )
