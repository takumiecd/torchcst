"""Periodic versions of existing grouped norm/patch/H-G contractions.

Packed fields match the existing 13-field matrix route. Bounds are unwrapped
start/high offsets; every pointer uses modular indices, without duplicate sites.
"""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _polar_atom, _raw, _store_param_cotangent


@tr.jit
def _index(j, N: tl.constexpr):
    return ((j % N) + N) % N


@tr.jit
def _periodic_raw(j, c, inv, O: tl.constexpr, L: tl.constexpr, N: tl.constexpr):
    delta = O + _index(j, N) * (L / N) - c
    delta = delta - L * tl.floor(tl.div_rn(delta, L) + 0.5)
    return _raw(delta, inv)


@tr.jit
def _span(c, inv, O: tl.constexpr, L: tl.constexpr, N: tl.constexpr):
    spacing: tl.constexpr = L / N
    radius = tl.sqrt_rn(tl.div_rn(1.0, inv))
    phase = c - O - L * tl.floor(tl.div_rn(c - O, L))
    error = 8.0 * 1.1920928955078125e-7 * (tl.abs(c) + tl.abs(O) + L + radius)
    precise = (
        (inv > 0.0) & (radius < float("inf")) & (error < spacing) & (radius < L / 2.0)
    )
    # Guard sites cover support-radius and coordinate/index rounding.
    low = tl.floor(tl.div_rn(phase - radius, spacing)).to(tl.int32) - 2
    high = tl.ceil(tl.div_rn(phase + radius, spacing)).to(tl.int32) + 3
    count = high - low
    precise = precise & (count < N)
    return tl.where(precise, low, 0), tl.where(precise, count, N)


@tr.jit
def _stats(
    c,
    inv,
    valid,
    O: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    SITES: tl.constexpr,
):
    low, count = _span(c, inv, O, L, N)
    count = tl.where(valid, count, 0)
    k = tl.arange(0, SITES)
    vv = tl.full((GROUP, SITES), 0.0, tl.float32)
    vd = tl.full((GROUP, SITES), 0.0, tl.float32)
    first, last = low + count, low
    live = tl.full((GROUP,), 0, tl.int32)
    for start in range(0, tl.max(count, 0), SITES):
        j = low[:, None] + start + k[None, :]
        v, dv = _periodic_raw(j, c[:, None], inv[:, None], O, L, N)
        mask = valid[:, None] & (j < (low + count)[:, None])
        v, dv = tl.where(mask, v, 0.0), tl.where(mask, dv, 0.0)
        vv += v * v
        vd += v * dv
        first = tl.minimum(first, tl.min(tl.where(v > 0, j, (low + count)[:, None]), 1))
        last = tl.maximum(last, tl.max(tl.where(v > 0, j + 1, low[:, None]), 1))
        live += tl.sum((v > 0).to(tl.int32), 1)
    # No full-site ID array is stored. A seam-crossing interval can contain
    # interior zeros on the full-axis fallback; evaluating those stays exact.
    last = tl.where(live > 0, last, first)
    return tl.sum(vv, 1), tl.sum(vd, 1), first, last, live


@tr.jit
def prepare(
    Source,
    P,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    Scalars,
    FLOOR: tl.constexpr,
    GROUP: tl.constexpr,
    SITES: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    amp, inv = _polar_atom(Source, tl.where(valid, a, 0), Scalars)
    co, ci = (
        tl.load(Source + 4 * a + 2, valid, 0),
        tl.load(Source + 4 * a + 3, valid, 0),
    )
    nv2, vd, ilo, ihi, iv = _stats(ci, inv, valid, OI, LI, NI, GROUP, SITES)
    nu2, ud, olo, ohi, ov = _stats(co, inv, valid, OO, LO, NO, GROUP, SITES)
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
def assemble(
    P,
    W,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    PATCH: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    amp, inv = tl.load(P + a, valid, 0), tl.load(P + A + a, valid, 1)
    ci, co = tl.load(P + 2 * A + a, valid, 0), tl.load(P + 3 * A + a, valid, 0)
    sv, su = tl.load(P + 4 * A + a, valid, 1), tl.load(P + 5 * A + a, valid, 1)
    ilo, ihi = (
        tl.load(P + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 10 * A + a, valid, 0).to(tl.int32),
    )
    olo, ohi = (
        tl.load(P + 11 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 12 * A + a, valid, 0).to(tl.int32),
    )
    k = tl.arange(0, PATCH)
    for istart in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), PATCH):
        i = olo[:, None] + istart + k[None, :]
        u, _ = _periodic_raw(i, co[:, None], inv[:, None], OO, LO, NO)
        u = tl.where(
            valid[:, None] & (i < ohi[:, None]), tl.div_rn(u, su[:, None]), 0.0
        )
        for jstart in range(0, tl.max(tl.maximum(ihi - ilo, 0), 0), PATCH):
            j = ilo[:, None] + jstart + k[None, :]
            v, _ = _periodic_raw(j, ci[:, None], inv[:, None], OI, LI, NI)
            v = tl.where(
                valid[:, None] & (j < ihi[:, None]), tl.div_rn(v, sv[:, None]), 0.0
            )
            mask = (
                valid[:, None, None]
                & (i[:, :, None] < ohi[:, None, None])
                & (j[:, None, :] < ihi[:, None, None])
            )
            tl.atomic_add(
                W + _index(i, NO)[:, :, None] * NI + _index(j, NI)[:, None, :],
                amp[:, None, None] * u[:, :, None] * v[:, None, :],
                mask,
                sem="relaxed",
            )


@tr.jit
def parameter_vjp(
    P,
    DW,
    DP,
    Source,
    AmplitudeMax,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    LI: tl.constexpr,
    LO: tl.constexpr,
    OI: tl.constexpr,
    OO: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    PATCH: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    amp, inv = tl.load(P + a, valid, 0), tl.load(P + A + a, valid, 1)
    ci, co = tl.load(P + 2 * A + a, valid, 0), tl.load(P + 3 * A + a, valid, 0)
    sv, su = tl.load(P + 4 * A + a, valid, 1), tl.load(P + 5 * A + a, valid, 1)
    gv, gu = tl.load(P + 6 * A + a, valid, 0), tl.load(P + 7 * A + a, valid, 0)
    flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    ilo, ihi = (
        tl.load(P + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 10 * A + a, valid, 0).to(tl.int32),
    )
    olo, ohi = (
        tl.load(P + 11 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 12 * A + a, valid, 0).to(tl.int32),
    )
    k = tl.arange(0, PATCH)
    da = tl.full((GROUP, PATCH, PATCH), 0.0, tl.float32)
    dco = tl.full((GROUP, PATCH, PATCH), 0.0, tl.float32)
    dci = tl.full((GROUP, PATCH, PATCH), 0.0, tl.float32)
    for istart in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), PATCH):
        i = olo[:, None] + istart + k[None, :]
        u, du = _periodic_raw(i, co[:, None], inv[:, None], OO, LO, NO)
        u = tl.div_rn(u, su[:, None])
        du = tl.where(
            (flags[:, None] & 2) != 0, 0.0, tl.div_rn(du, su[:, None]) - gu[:, None] * u
        )
        mask_i = valid[:, None] & (i < ohi[:, None])
        u, du = tl.where(mask_i, u, 0.0), tl.where(mask_i, du, 0.0)
        for jstart in range(0, tl.max(tl.maximum(ihi - ilo, 0), 0), PATCH):
            j = ilo[:, None] + jstart + k[None, :]
            v, dv = _periodic_raw(j, ci[:, None], inv[:, None], OI, LI, NI)
            v = tl.div_rn(v, sv[:, None])
            dv = tl.where(
                (flags[:, None] & 1) != 0,
                0.0,
                tl.div_rn(dv, sv[:, None]) - gv[:, None] * v,
            )
            mask_j = valid[:, None] & (j < ihi[:, None])
            v, dv = tl.where(mask_j, v, 0.0), tl.where(mask_j, dv, 0.0)
            mask = mask_i[:, :, None] & mask_j[:, None, :]
            dw = tl.load(
                DW + _index(i, NO)[:, :, None] * D0 + _index(j, NI)[:, None, :] * D1,
                mask,
                0.0,
            )
            da += dw * u[:, :, None] * v[:, None, :]
            dco += dw * du[:, :, None] * v[:, None, :]
            dci += dw * u[:, :, None] * dv[:, None, :]
    _store_param_cotangent(
        DP,
        Source,
        AmplitudeMax,
        a,
        valid,
        tl.sum(tl.sum(da, 2), 1),
        amp * tl.sum(tl.sum(dci, 2), 1),
        amp * tl.sum(tl.sum(dco, 2), 1),
        True,
        CENTER_OUTPUT_FIRST=True,
    )


@tr.jit
def axis_contract(
    X,
    P,
    T,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    OUTPUT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    inv = tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    other_low = tl.load(P + (9 if OUTPUT else 11) * A + a, valid, 0)
    other_high = tl.load(P + (10 if OUTPUT else 12) * A + a, valid, 0)
    high = tl.where(other_high > other_low, high, low)
    acc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(high - low, 0), 0), BK):
        j = low[:, None] + start + k[None, :]
        v, _ = _periodic_raw(j, center[:, None], inv[:, None], O, L, N)
        v = tl.div_rn(v, norm[:, None])
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < high[:, None, None])
        )
        x = tl.load(
            X + rows[None, :, None] * X0 + _index(j, N)[:, None, :] * X1, mask, 0.0
        )
        acc += x * v[:, None, :]
    tl.store(
        T + a[:, None] * B + rows[None, :],
        tl.sum(acc, 2),
        valid[:, None] & (rows[None, :] < B),
    )


@tr.jit
def axis_scatter(
    T,
    P,
    Y,
    A: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    OUTPUT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    amp, inv = tl.load(P + a, valid, 0), tl.load(P + A + a, valid, 1)
    center = tl.load(P + (3 if OUTPUT else 2) * A + a, valid, 0)
    norm = tl.load(P + (5 if OUTPUT else 4) * A + a, valid, 1)
    low = tl.load(P + (11 if OUTPUT else 9) * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + (12 if OUTPUT else 10) * A + a, valid, 0).to(tl.int32)
    t = tl.load(
        T + a[:, None] * B + rows[None, :], valid[:, None] & (rows[None, :] < B), 0.0
    )
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
            t[:, :, None] * v[:, None, :],
            mask,
            sem="relaxed",
        )


@tr.jit
def factor_vjp(
    X,
    DY,
    P,
    H,
    G,
    DP,
    Source,
    AmplitudeMax,
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
    BM: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
):
    a = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    valid = a < A
    rows, k = tl.arange(0, BM), tl.arange(0, BK)
    amp, inv = tl.load(P + a, valid, 0), tl.load(P + A + a, valid, 1)
    ci, co = tl.load(P + 2 * A + a, valid, 0), tl.load(P + 3 * A + a, valid, 0)
    sv, su = tl.load(P + 4 * A + a, valid, 1), tl.load(P + 5 * A + a, valid, 1)
    gv, gu = tl.load(P + 6 * A + a, valid, 0), tl.load(P + 7 * A + a, valid, 0)
    flags = tl.load(P + 8 * A + a, valid, 0).to(tl.int32)
    ilo, ihi = (
        tl.load(P + 9 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 10 * A + a, valid, 0).to(tl.int32),
    )
    olo, ohi = (
        tl.load(P + 11 * A + a, valid, 0).to(tl.int32),
        tl.load(P + 12 * A + a, valid, 0).to(tl.int32),
    )
    h = tl.load(
        H + a[:, None] * B + rows[None, :], valid[:, None] & (rows[None, :] < B), 0.0
    )
    g = tl.load(
        G + a[:, None] * B + rows[None, :], valid[:, None] & (rows[None, :] < B), 0.0
    )
    dc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), BK):
        i = olo[:, None] + start + k[None, :]
        u, du = _periodic_raw(i, co[:, None], inv[:, None], OO, LO, NO)
        du = tl.where(
            (flags[:, None] & 2) != 0,
            0.0,
            tl.div_rn(du, su[:, None]) - gu[:, None] * tl.div_rn(u, su[:, None]),
        )
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (i[:, None, :] < ohi[:, None, None])
        )
        dy = tl.load(
            DY + rows[None, :, None] * D0 + _index(i, NO)[:, None, :] * D1, mask, 0.0
        )
        dc += dy * du[:, None, :]
    dco = amp * tl.sum(h * tl.sum(dc, 2), 1)
    dc = tl.full((GROUP, BM, BK), 0.0, tl.float32)
    for start in range(0, tl.max(tl.maximum(ihi - ilo, 0), 0), BK):
        j = ilo[:, None] + start + k[None, :]
        v, dv = _periodic_raw(j, ci[:, None], inv[:, None], OI, LI, NI)
        dv = tl.where(
            (flags[:, None] & 1) != 0,
            0.0,
            tl.div_rn(dv, sv[:, None]) - gv[:, None] * tl.div_rn(v, sv[:, None]),
        )
        mask = (
            valid[:, None, None]
            & (rows[None, :, None] < B)
            & (j[:, None, :] < ihi[:, None, None])
        )
        x = tl.load(
            X + rows[None, :, None] * X0 + _index(j, NI)[:, None, :] * X1, mask, 0.0
        )
        dc += x * dv[:, None, :]
    dci = amp * tl.sum(g * tl.sum(dc, 2), 1)
    da = tl.sum(h * g, 1)
    _store_param_cotangent(
        DP, Source, AmplitudeMax, a, valid, da, dci, dco, True, CENTER_OUTPUT_FIRST=True
    )
