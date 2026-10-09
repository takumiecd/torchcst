"""Exact bounded trigonometry and intrinsic centre-fibre profile derivatives."""

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
