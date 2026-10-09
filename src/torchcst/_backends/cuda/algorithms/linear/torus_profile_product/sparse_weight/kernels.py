"""Exact sparse centre-fibre factors and shared W/dW contractions.

Circle window is a conservative index interval on regular physical Strip sites.
Unsafe windows and support-capacity overflow take complete-axis paths.
"""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .._shared.profiles import _atom, _circle, _section


@tr.jit
def pack(
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    Index,
    Raw,
    Stats,
    Start,
    Spacing,
    Pitch,
    N: tl.constexpr,
    CAP: tl.constexpr,
    V: tl.constexpr,
    TILE: tl.constexpr,
    SIDE: tl.constexpr,
):
    a = tl.program_id(0)
    (
        amp,
        inv,
        major,
        minor,
        arc,
        q0,
        q1,
        q2,
        w1,
        w2,
        rho2,
        sinc,
        z0,
        z1,
        zr,
        maximum,
        r2,
    ) = _atom(P, Prec, Maximum, Major, Minor, "bounded-poly")
    if SIDE == 0:
        spacing, pitch, start = tl.load(Spacing), tl.load(Pitch), tl.load(Start)
        period = major.to(tl.float64) * 6.283185307179586
        relative = arc.to(tl.float64) - start.to(tl.float64)
        physical = relative - tl.floor(relative / period) * period
        tile = tl.floor(physical / pitch.to(tl.float64))
        centre = (tile * TILE + tl.floor((physical - tile * pitch) / spacing)).to(
            tl.int32
        )
        radius = major + minor * q0
        ratio = libdevice.sqrt(tl.div_rn(1.0, inv)) / (2 * tl.abs(radius))
        halfspan = 2 * major * libdevice.asin(tl.minimum(ratio, 1.0))
        # Mapping a physical gap to an index can shift the candidate centre by
        # at most pitch/spacing - TILE + 1. Two extra indices cover rounding.
        bound = halfspan / spacing + pitch / spacing - TILE + 2
        window = (
            (N >= CAP)
            & (tl.abs(period - (N // TILE) * pitch) < spacing)
            & (spacing > 0)
            & (pitch >= TILE * spacing)
            & (ratio < 1)
            & (N % TILE == 0)
            & (bound < CAP / 2 - 1)
        )
        if window:
            k = tl.arange(0, CAP)
            idx = ((centre + k - CAP // 2) % N + N) % N
            u, d0, d1 = _circle(
                Circle, idx, N, arc, major, minor, q0, inv, "bounded-poly"
            )
            present = u > 0
            rank = tl.cumsum(present.to(tl.int32)) - 1
            count = tl.sum(present.to(tl.int32))
            tl.store(Index + a * CAP + rank, idx, present)
            tl.store(Raw + a * CAP + rank, u, present)
            tl.store(Stats + a * 5, count)
            tl.store(Stats + a * 5 + 1, tl.sum(u * u))
            tl.store(Stats + a * 5 + 2, tl.sum(u * d0))
            tl.store(Stats + a * 5 + 3, tl.sum(u * d1))
            tl.store(Stats + a * 5 + 4, 0.0)
        else:
            full_idx = tl.arange(0, V)
            full_u, full_d0, full_d1 = _circle(
                Circle, full_idx, N, arc, major, minor, q0, inv, "bounded-poly"
            )
            full_present = (full_idx < N) & (full_u > 0)
            full_rank = tl.cumsum(full_present.to(tl.int32)) - 1
            full_count = tl.sum(full_present.to(tl.int32))
            full_mask = full_present & (full_count <= CAP)
            tl.store(Index + a * CAP + full_rank, full_idx, full_mask)
            tl.store(Raw + a * CAP + full_rank, full_u, full_mask)
            tl.store(Stats + a * 5, full_count)
            tl.store(Stats + a * 5 + 1, tl.sum(full_u * full_u))
            tl.store(Stats + a * 5 + 2, tl.sum(full_u * full_d0))
            tl.store(Stats + a * 5 + 3, tl.sum(full_u * full_d1))
            tl.store(Stats + a * 5 + 4, 0.0)
    else:
        idx = tl.arange(0, V)
        v, d0, d1, d2 = _section(Sites, idx, N, minor, q0, q1, q2, inv)
        present = (idx < N) & (v > 0)
        rank = tl.cumsum(present.to(tl.int32)) - 1
        count = tl.sum(present.to(tl.int32))
        mask = present & (count <= CAP)
        tl.store(Index + a * CAP + rank, idx, mask)
        tl.store(Raw + a * CAP + rank, v, mask)
        tl.store(Stats + a * 5, count)
        tl.store(Stats + a * 5 + 1, tl.sum(v * v))
        tl.store(Stats + a * 5 + 2, tl.sum(v * d0))
        tl.store(Stats + a * 5 + 3, tl.sum(v * d1))
        tl.store(Stats + a * 5 + 4, tl.sum(v * d2))


@tr.jit
def block(
    Index,
    Raw,
    a,
    base,
    Count,
    N: tl.constexpr,
    CAP: tl.constexpr,
    T: tl.constexpr,
    FULL: tl.constexpr,
):
    off = base + tl.arange(0, T)
    if FULL:
        idx, valid = off, off < N
        raw = tl.full((T,), 0, tl.float32)
    else:
        valid = off < Count
        idx = tl.load(Index + a * CAP + off, valid, 0)
        raw = tl.load(Raw + a * CAP + off, valid, 0)
    valid = valid & (idx >= 0) & (idx < N)
    idx = tl.where(valid, idx, 0)
    return idx, valid, raw


@tr.jit
def patches(
    W,
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    OI,
    OR,
    OS,
    II,
    IR,
    IS,
    NI: tl.constexpr,
    NO: tl.constexpr,
    CO: tl.constexpr,
    CI: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
    VJP: tl.constexpr,
    DP,
    FULL_O: tl.constexpr,
    FULL_I: tl.constexpr,
):
    a = tl.program_id(0)
    (
        amp,
        inv,
        major,
        minor,
        arc,
        q0,
        q1,
        q2,
        w1,
        w2,
        rho2,
        sinc,
        z0,
        z1,
        zr,
        maximum,
        r2,
    ) = _atom(P, Prec, Maximum, Major, Minor, "bounded-poly")
    oc, ic = tl.load(OS + a * 5), tl.load(IS + a * 5)
    un, vn = tl.load(OS + a * 5 + 1), tl.load(IS + a * 5 + 1)
    norm = libdevice.sqrt(un * vn)
    denom = tl.maximum(norm, FLOOR)
    scale = tl.div_rn(amp, denom)
    dot = tl.full((), 0, tl.float32)
    darc = tl.full((), 0, tl.float32)
    dq0 = tl.full((), 0, tl.float32)
    dq1 = tl.full((), 0, tl.float32)
    dq2 = tl.full((), 0, tl.float32)
    for ob in range(0, NO if FULL_O else CO, T):
        if FULL_O or ob < oc:
            oi, om, u = block(OI, OR, a, ob, oc, NO, CO, T, FULL_O)
            if FULL_O or VJP:
                uu, ua, uq = _circle(
                    Circle, oi, NO, arc, major, minor, q0, inv, "bounded-poly"
                )
                if FULL_O:
                    u = tl.where(om, uu, 0.0)
            for ib in range(0, NI if FULL_I else CI, T):
                if FULL_I or ib < ic:
                    ii, im, v = block(II, IR, a, ib, ic, NI, CI, T, FULL_I)
                    if FULL_I or VJP:
                        vv, v0, v1, v2 = _section(Sites, ii, NI, minor, q0, q1, q2, inv)
                        if FULL_I:
                            v = tl.where(im, vv, 0.0)
                    mask = (
                        om[:, None]
                        & im[None, :]
                        & (u[:, None] != 0)
                        & (v[None, :] != 0)
                    )
                    address = W + oi[:, None] * NI + ii[None, :]
                    if VJP:
                        dw = tl.load(address, mask, 0.0)
                        ru = tl.sum(dw * v[None, :], 1)
                        rv = tl.sum(dw * u[:, None], 0)
                        dot += tl.sum(ru * u)
                        darc += tl.sum(ru * ua)
                        dq0 += tl.sum(ru * uq) + tl.sum(rv * v0)
                        dq1 += tl.sum(rv * v1)
                        dq2 += tl.sum(rv * v2)
                    else:
                        tl.atomic_add(
                            address,
                            scale * u[:, None] * v[None, :],
                            mask,
                            sem="relaxed",
                        )
    if VJP:
        correction = tl.where(norm >= FLOOR, dot, 0.0)
        safe_u, safe_v = tl.where(un > 0, un, 1.0), tl.where(vn > 0, vn, 1.0)
        darc = scale * (darc - correction * tl.div_rn(tl.load(OS + a * 5 + 2), safe_u))
        dq0 = scale * (
            dq0
            - correction
            * (
                tl.div_rn(tl.load(OS + a * 5 + 3), safe_u)
                + tl.div_rn(tl.load(IS + a * 5 + 2), safe_v)
            )
        )
        dq1 = scale * (dq1 - correction * tl.div_rn(tl.load(IS + a * 5 + 3), safe_v))
        dq2 = scale * (dq2 - correction * tl.div_rn(tl.load(IS + a * 5 + 4), safe_v))
        coeff = tl.where(
            rho2 < 1e-4,
            -1.0 / 3.0 + rho2 / 30.0 - rho2 * rho2 / 840.0,
            tl.div_rn(q0 - sinc, tl.where(rho2 > 0, rho2, 1.0)),
        )
        tangent = w1 * dq1 + w2 * dq2
        ds1 = tl.div_rn(-sinc * w1 * dq0 + sinc * dq1 + coeff * w1 * tangent, minor)
        ds2 = tl.div_rn(-sinc * w2 * dq0 + sinc * dq2 + coeff * w2 * tangent, minor)
        a0, a1 = tl.div_rn(z0, zr), tl.div_rn(z1, zr)
        active = (r2 >= 1.1754943508222875e-38).to(tl.float32)
        ampgrad = tl.div_rn(dot, denom) * tl.div_rn(maximum, zr)
        tl.store(DP + 5 * a, ampgrad * (1 - active * a0 * a0))
        tl.store(DP + 5 * a + 1, -ampgrad * active * a0 * a1)
        tl.store(DP + 5 * a + 2, darc)
        tl.store(DP + 5 * a + 3, ds1)
        tl.store(DP + 5 * a + 4, ds2)


@tr.jit
def contract(
    W,
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    OI,
    OR,
    OS,
    II,
    IR,
    IS,
    DP,
    NI: tl.constexpr,
    NO: tl.constexpr,
    CO: tl.constexpr,
    CI: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
    VJP: tl.constexpr,
):
    a = tl.program_id(0)
    full_o = tl.load(OS + a * 5) > CO
    full_i = tl.load(IS + a * 5) > CI
    if full_o:
        if full_i:
            patches(
                W,
                P,
                Prec,
                Maximum,
                Major,
                Minor,
                Circle,
                Sites,
                OI,
                OR,
                OS,
                II,
                IR,
                IS,
                NI,
                NO,
                CO,
                CI,
                T,
                FLOOR,
                VJP,
                DP,
                True,
                True,
            )
        else:
            patches(
                W,
                P,
                Prec,
                Maximum,
                Major,
                Minor,
                Circle,
                Sites,
                OI,
                OR,
                OS,
                II,
                IR,
                IS,
                NI,
                NO,
                CO,
                CI,
                T,
                FLOOR,
                VJP,
                DP,
                True,
                False,
            )
    else:
        if full_i:
            patches(
                W,
                P,
                Prec,
                Maximum,
                Major,
                Minor,
                Circle,
                Sites,
                OI,
                OR,
                OS,
                II,
                IR,
                IS,
                NI,
                NO,
                CO,
                CI,
                T,
                FLOOR,
                VJP,
                DP,
                False,
                True,
            )
        else:
            patches(
                W,
                P,
                Prec,
                Maximum,
                Major,
                Minor,
                Circle,
                Sites,
                OI,
                OR,
                OS,
                II,
                IR,
                IS,
                NI,
                NO,
                CO,
                CI,
                T,
                FLOOR,
                VJP,
                DP,
                False,
                False,
            )
