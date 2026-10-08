"""Exact full-axis traversal. Each CTA owns one atom and all batch contractions.

Optional saved H/input-VJP contractions and per-atom geometry/norms cross the
forward/backward boundary. G and current output contractions remain CTA-local.
No factor arrays or weight/weight-gradient buffers exist. Only Y/dX are atomic;
dP is stored once per atom. Compiler spills must be
checked on each target before describing the intermediates as on-chip.
"""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _sin_bounded(x, TRIG: tl.constexpr):
    if TRIG == "bounded-poly":
        # Full valid interval [-pi, pi]; FP64 Horner, one FP32 output rounding.
        xx = x.to(tl.float64)
        z = xx * xx
        p = tl.full((), -3.868170170630684e-23, tl.float64)
        p = p * z + (1.9572941063391263e-20)
        p = p * z + (-8.22063524662433e-18)
        p = p * z + (2.8114572543455206e-15)
        p = p * z + (-7.647163731819816e-13)
        p = p * z + (1.6059043836821613e-10)
        p = p * z + (-2.505210838544172e-08)
        p = p * z + (2.7557319223985893e-06)
        p = p * z + (-0.0001984126984126984)
        p = p * z + (0.008333333333333333)
        p = p * z + (-0.16666666666666666)
        p = p * z + (1.0)
        return (p * xx).to(tl.float32)
    if TRIG == "fp64":
        return libdevice.sin(x.to(tl.float64)).to(tl.float32)
    # FP32 hardware sine: all callers supply valid angles within +/- pi.
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _cos_bounded(x, TRIG: tl.constexpr):
    if TRIG == "bounded-poly":
        # Full valid interval [-pi, pi]; FP64 Horner, one FP32 output rounding.
        xx = x.to(tl.float64)
        z = xx * xx
        p = tl.full((), 1.6117375710961184e-24, tl.float64)
        p = p * z + (-8.896791392450574e-22)
        p = p * z + (4.110317623312165e-19)
        p = p * z + (-1.5619206968586225e-16)
        p = p * z + (4.779477332387385e-14)
        p = p * z + (-1.1470745597729725e-11)
        p = p * z + (2.08767569878681e-09)
        p = p * z + (-2.755731922398589e-07)
        p = p * z + (2.48015873015873e-05)
        p = p * z + (-0.001388888888888889)
        p = p * z + (0.041666666666666664)
        p = p * z + (-0.5)
        p = p * z + (1.0)
        return p.to(tl.float32)
    if TRIG == "fp64":
        return libdevice.cos(x.to(tl.float64)).to(tl.float32)
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@tr.jit
def _atom(P, Prec, Maximum, Major, Minor, TRIG: tl.constexpr):
    a = tl.program_id(0)
    z0, z1 = tl.load(P + 5 * a), tl.load(P + 5 * a + 1)
    arc = tl.load(P + 5 * a + 2)
    s1, s2 = tl.load(P + 5 * a + 3), tl.load(P + 5 * a + 4)
    major, minor = tl.load(Major), tl.load(Minor)
    w1, w2 = tl.div_rn(s1, minor), tl.div_rn(s2, minor)
    rho2 = w1 * w1 + w2 * w2
    rho = libdevice.sqrt(rho2)
    # Valid intrinsic section coordinates have rho <= pi. Bounded sin/cos
    # avoid libdevice's unneeded huge-angle slow path and its local array.
    safe = tl.where(rho > 0, rho, 1.0)
    sinc = tl.where(rho > 0, tl.div_rn(_sin_bounded(rho, TRIG), safe), 1.0)
    q0, q1, q2 = _cos_bounded(rho, TRIG), sinc * w1, sinc * w2
    r2 = z0 * z0 + z1 * z1
    zr = libdevice.sqrt(tl.maximum(r2, 1.1754943508222875e-38))
    maximum = tl.load(Maximum)
    amp = maximum * tl.div_rn(z0, zr)
    return (
        amp,
        tl.load(Prec + a),
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
    )


@tr.jit
def _circle(
    Circle, i, NO: tl.constexpr, arc, major, minor, q0, inv, TRIG: tl.constexpr
):
    theta = tl.load(Circle + i, i < NO, other=0)
    delta = theta - arc.to(tl.float64) / major.to(tl.float64)
    delta = delta + 3.141592653589793
    delta = delta - tl.floor(delta / 6.283185307179586) * 6.283185307179586
    delta = (delta - 3.141592653589793).to(tl.float32)
    radius = major + minor * q0
    # Delta was reduced in FP64 to [-pi, pi); both sine arguments are bounded.
    sh = _sin_bounded(delta / 2, TRIG)
    squared = 4 * radius * radius * sh * sh
    gap = tl.maximum(1 - squared * inv, 0)
    raw = tl.where(i < NO, gap * gap * gap, 0)
    deriv = tl.where(i < NO, -3 * gap * gap * inv, 0)
    da = deriv * (-2 * radius * radius * _sin_bounded(delta, TRIG) / major)
    dq0 = deriv * (8 * radius * minor * sh * sh)
    return raw, da, dq0


@tr.jit
def _section(Sites, j, NI: tl.constexpr, minor, q0, q1, q2, inv):
    d0 = q0 - tl.load(Sites + 3 * j, j < NI, other=0)
    d1 = q1 - tl.load(Sites + 3 * j + 1, j < NI, other=0)
    d2 = q2 - tl.load(Sites + 3 * j + 2, j < NI, other=0)
    squared = minor * minor * (d0 * d0 + d1 * d1 + d2 * d2)
    gap = tl.maximum(1 - squared * inv, 0)
    raw = tl.where(j < NI, gap * gap * gap, 0)
    derivative = tl.where(j < NI, -3 * gap * gap * inv * 2 * minor * minor, 0)
    return raw, derivative * d0, derivative * d1, derivative * d2


@tr.jit
def forward(
    X,
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    HCache,
    Info,
    Y,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
    TRIG: tl.constexpr = "bounded-poly",
    SAVE: tl.constexpr = "recompute",
):
    (
        amp,
        inv,
        major,
        minor,
        arc,
        q0,
        q1,
        q2,
        _w1,
        _w2,
        _rho2,
        _sinc,
        _z0,
        _z1,
        _zr,
        _maximum,
        _r2,
    ) = _atom(P, Prec, Maximum, Major, Minor, TRIG)
    b, t = tl.arange(0, PB), tl.arange(0, T)
    h = tl.full((PB,), 0, tl.float32)
    if SAVE == "vjp":
        h0, h1, h2 = h, h, h
        nv0 = tl.full((), 0, tl.float32)
        nv1 = tl.full((), 0, tl.float32)
        nv2d = tl.full((), 0, tl.float32)
        nua = tl.full((), 0, tl.float32)
        nu0 = tl.full((), 0, tl.float32)
    nv2 = tl.full((), 0, tl.float32)
    nu2 = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(NI, T)):
        j = start * T + t
        v, dv0, dv1, dv2 = _section(Sites, j, NI, minor, q0, q1, q2, inv)
        xx = tl.load(
            X + b[:, None] * NI + j[None, :],
            (b[:, None] < B) & (j[None, :] < NI),
            other=0,
        )
        h += tl.sum(xx * v[None, :], 1)
        nv2 += tl.sum(v * v, 0)
        if SAVE == "vjp":
            h0 += tl.sum(xx * dv0[None, :], 1)
            h1 += tl.sum(xx * dv1[None, :], 1)
            h2 += tl.sum(xx * dv2[None, :], 1)
            nv0 += tl.sum(v * dv0, 0)
            nv1 += tl.sum(v * dv1, 0)
            nv2d += tl.sum(v * dv2, 0)
    for start in range(tl.cdiv(NO, T)):
        i = start * T + t
        u, duarc, duq0 = _circle(Circle, i, NO, arc, major, minor, q0, inv, TRIG)
        nu2 += tl.sum(u * u, 0)
        if SAVE == "vjp":
            nua += tl.sum(u * duarc, 0)
            nu0 += tl.sum(u * duq0, 0)
    if SAVE != "recompute":
        a = tl.program_id(0)
        C: tl.constexpr = 11 if SAVE == "vjp" else 6
        K: tl.constexpr = 4 if SAVE == "vjp" else 1
        if SAVE != "norm":
            tl.store(HCache + a * K * B + b, h, b < B)
        tl.store(Info + a * C, q0)
        tl.store(Info + a * C + 1, q1)
        tl.store(Info + a * C + 2, q2)
        tl.store(Info + a * C + 3, _sinc)
        tl.store(Info + a * C + 4, nu2)
        tl.store(Info + a * C + 5, nv2)
        if SAVE == "vjp":
            tl.store(HCache + (a * K + 1) * B + b, h0, b < B)
            tl.store(HCache + (a * K + 2) * B + b, h1, b < B)
            tl.store(HCache + (a * K + 3) * B + b, h2, b < B)
            tl.store(Info + a * C + 6, nua)
            tl.store(Info + a * C + 7, nu0)
            tl.store(Info + a * C + 8, nv0)
            tl.store(Info + a * C + 9, nv1)
            tl.store(Info + a * C + 10, nv2d)
    norm = libdevice.sqrt(nu2) * libdevice.sqrt(nv2)
    scale = tl.div_rn(amp, tl.maximum(norm, FLOOR))
    for start in range(tl.cdiv(NO, T)):
        i = start * T + t
        u, _duarc, _duq0 = _circle(Circle, i, NO, arc, major, minor, q0, inv, TRIG)
        mask = (b[:, None] < B) & (i[None, :] < NO) & (u[None, :] != 0)
        tl.atomic_add(
            Y + b[:, None] * NO + i[None, :],
            scale * h[:, None] * u[None, :],
            mask,
            sem="relaxed",
        )


@tr.jit
def backward(
    X,
    DY,
    P,
    Prec,
    Maximum,
    Major,
    Minor,
    Circle,
    Sites,
    HCache,
    Info,
    DX,
    DP,
    B: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    PB: tl.constexpr,
    T: tl.constexpr,
    FLOOR: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_P: tl.constexpr,
    TRIG: tl.constexpr = "bounded-poly",
    SAVE: tl.constexpr = "recompute",
):
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
    ) = _atom(P, Prec, Maximum, Major, Minor, TRIG)
    b, t = tl.arange(0, PB), tl.arange(0, T)
    h, h0, h1, h2 = (
        tl.full((PB,), 0, tl.float32),
        tl.full((PB,), 0, tl.float32),
        tl.full((PB,), 0, tl.float32),
        tl.full((PB,), 0, tl.float32),
    )
    g, ga, g0 = (
        tl.full((PB,), 0, tl.float32),
        tl.full((PB,), 0, tl.float32),
        tl.full((PB,), 0, tl.float32),
    )
    nv2, nv0, nv1, nv2d = (
        tl.full((), 0, tl.float32),
        tl.full((), 0, tl.float32),
        tl.full((), 0, tl.float32),
        tl.full((), 0, tl.float32),
    )
    nu2, nua, nu0 = (
        tl.full((), 0, tl.float32),
        tl.full((), 0, tl.float32),
        tl.full((), 0, tl.float32),
    )
    if SAVE != "recompute":
        a = tl.program_id(0)
        C: tl.constexpr = 11 if SAVE == "vjp" else 6
        K: tl.constexpr = 4 if SAVE == "vjp" else 1
        q0 = tl.load(Info + a * C)
        q1 = tl.load(Info + a * C + 1)
        q2 = tl.load(Info + a * C + 2)
        sinc = tl.load(Info + a * C + 3)
        nu2 = tl.load(Info + a * C + 4)
        nv2 = tl.load(Info + a * C + 5)
        if NEED_P:
            h = tl.load(HCache + a * K * B + b, b < B, other=0)
            if SAVE == "vjp":
                h0 = tl.load(HCache + (a * K + 1) * B + b, b < B, other=0)
                h1 = tl.load(HCache + (a * K + 2) * B + b, b < B, other=0)
                h2 = tl.load(HCache + (a * K + 3) * B + b, b < B, other=0)
                nua = tl.load(Info + a * C + 6)
                nu0 = tl.load(Info + a * C + 7)
                nv0 = tl.load(Info + a * C + 8)
                nv1 = tl.load(Info + a * C + 9)
                nv2d = tl.load(Info + a * C + 10)
    if SAVE == "recompute" or (NEED_P and SAVE == "h"):
        for start in range(tl.cdiv(NI, T)):
            j = start * T + t
            v, dv0, dv1, dv2 = _section(Sites, j, NI, minor, q0, q1, q2, inv)
            if SAVE == "recompute":
                nv2 += tl.sum(v * v, 0)
            if NEED_P:
                xx = tl.load(
                    X + b[:, None] * NI + j[None, :],
                    (b[:, None] < B) & (j[None, :] < NI),
                    other=0,
                )
                if SAVE == "recompute":
                    h += tl.sum(xx * v[None, :], 1)
                h0 += tl.sum(xx * dv0[None, :], 1)
                h1 += tl.sum(xx * dv1[None, :], 1)
                h2 += tl.sum(xx * dv2[None, :], 1)
                nv0 += tl.sum(v * dv0, 0)
                nv1 += tl.sum(v * dv1, 0)
                nv2d += tl.sum(v * dv2, 0)
    for start in range(tl.cdiv(NO, T)):
        i = start * T + t
        u, dua, du0 = _circle(Circle, i, NO, arc, major, minor, q0, inv, TRIG)
        yy = tl.load(
            DY + b[:, None] * NO + i[None, :],
            (b[:, None] < B) & (i[None, :] < NO),
            other=0,
        )
        g += tl.sum(yy * u[None, :], 1)
        if SAVE == "recompute":
            nu2 += tl.sum(u * u, 0)
        if NEED_P:
            ga += tl.sum(yy * dua[None, :], 1)
            g0 += tl.sum(yy * du0[None, :], 1)
            if SAVE != "vjp":
                nua += tl.sum(u * dua, 0)
                nu0 += tl.sum(u * du0, 0)
    norm = libdevice.sqrt(nu2) * libdevice.sqrt(nv2)
    denom = tl.maximum(norm, FLOOR)
    scale = tl.div_rn(amp, denom)
    if NEED_X:
        for start in range(tl.cdiv(NI, T)):
            j = start * T + t
            v, dv0, dv1, dv2 = _section(Sites, j, NI, minor, q0, q1, q2, inv)
            mask = (b[:, None] < B) & (j[None, :] < NI) & (v[None, :] != 0)
            tl.atomic_add(
                DX + b[:, None] * NI + j[None, :],
                scale * g[:, None] * v[None, :],
                mask,
                sem="relaxed",
            )
    if NEED_P:
        dot = tl.sum(h * g, 0)
        correction = tl.where(norm >= FLOOR, dot, 0)
        # Empty norm has zero derivative; safe denominators avoid 0/0.
        un = tl.where(nu2 > 0, nu2, 1)
        vn = tl.where(nv2 > 0, nv2, 1)
        darc = scale * (tl.sum(h * ga, 0) - correction * tl.div_rn(nua, un))
        dq0 = scale * (
            tl.sum(h * g0 + h0 * g, 0)
            - correction * (tl.div_rn(nu0, un) + tl.div_rn(nv0, vn))
        )
        dq1 = scale * (tl.sum(h1 * g, 0) - correction * tl.div_rn(nv1, vn))
        dq2 = scale * (tl.sum(h2 * g, 0) - correction * tl.div_rn(nv2d, vn))
        # Jacobian of normal-coordinate expmap on S², with analytic origin limit.
        denomrho = tl.where(rho2 > 0, rho2, 1)
        coeff = tl.where(
            rho2 < 1e-4,
            -1.0 / 3.0 + rho2 / 30.0 - rho2 * rho2 / 840.0,
            tl.div_rn(q0 - sinc, denomrho),
        )
        tangent = w1 * dq1 + w2 * dq2
        ds1 = tl.div_rn(-sinc * w1 * dq0 + sinc * dq1 + coeff * w1 * tangent, minor)
        ds2 = tl.div_rn(-sinc * w2 * dq0 + sinc * dq2 + coeff * w2 * tangent, minor)
        a0, a1 = tl.div_rn(z0, zr), tl.div_rn(z1, zr)
        active = (r2 >= 1.1754943508222875e-38).to(tl.float32)
        ampgrad = tl.div_rn(dot, denom) * tl.div_rn(maximum, zr)
        a = tl.program_id(0)
        tl.store(DP + 5 * a, ampgrad * (1 - active * a0 * a0))
        tl.store(DP + 5 * a + 1, -ampgrad * active * a0 * a1)
        tl.store(DP + 5 * a + 2, darc)
        tl.store(DP + 5 * a + 3, ds1)
        tl.store(DP + 5 * a + 4, ds2)
