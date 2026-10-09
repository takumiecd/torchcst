"""Live Polar decode and exact intrinsic-S² expmap/Jacobian snapshots."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def normalize_sites(Sites, Radius, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Sites + 3 * i, i < N, other=0)
    y = tl.load(Sites + 3 * i + 1, i < N, other=0)
    z = tl.load(Sites + 3 * i + 2, i < N, other=0)
    norm = libdevice.sqrt(x * x + y * y + z * z)
    scale = tl.div_rn(tl.load(Radius).to(tl.float32), norm)
    tl.store(Out + 3 * i, x * scale, i < N)
    tl.store(Out + 3 * i + 1, y * scale, i < N)
    tl.store(Out + 3 * i + 2, z * scale, i < N)


@tr.jit
def _precision(amp, alpha, Shared, Bounds):
    minimum = tl.load(Bounds[0]).to(tl.float32)
    birth = tl.load(Bounds[1]).to(tl.float32)
    maximum = tl.load(Bounds[2]).to(tl.float32)
    ratio = tl.div_rn(amp, tl.load(Shared[1]).to(tl.float32))
    x = ratio * ratio
    kappa = tl.load(Shared[2]).to(tl.float32)
    upper_x = libdevice.pow(x, tl.load(Shared[4]).to(tl.float32))
    upper = minimum + tl.div_rn((maximum - minimum) * kappa, kappa + upper_x)
    upper = tl.maximum(upper, tl.load(Bounds[3]).to(tl.float32))
    lower = minimum + tl.div_rn(
        birth - minimum, 1.0 + tl.load(Shared[3]).to(tl.float32) * x
    )
    upper = tl.maximum(upper, lower)
    sigma = libdevice.exp(
        (1.0 - alpha) * libdevice.log(lower) + alpha * libdevice.log(upper)
    )
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    inv = tl.div_rn(1.0, sigma)
    return inv * inv


@tr.jit
def _geometry(Source, a, valid, Radius, Q, Jac, COL: tl.constexpr):
    c0 = tl.load(Source + 6 * a + COL, valid, other=0)
    c1 = tl.load(Source + 6 * a + COL + 1, valid, other=0)
    r = tl.load(Radius).to(tl.float32)
    radial2 = c0 * c0 + c1 * c1
    radial = libdevice.sqrt(radial2)
    theta = tl.div_rn(radial, r)
    # Torch sinc(theta/pi) multiplies its argument by pi inside the op.
    sinc_arg = tl.div_rn(theta, 3.141592653589793) * 3.141592653589793
    scale = tl.where(sinc_arg == 0.0, 1.0, tl.div_rn(libdevice.sin(sinc_arg), sinc_arg))
    cosine = libdevice.cos(theta)
    t2 = theta * theta
    curvature = tl.where(
        tl.abs(theta) < 1e-3,
        tl.div_rn(-1.0 / 3.0 + t2 / 30.0 - t2 * t2 / 840.0, r * r),
        tl.div_rn(cosine - scale, tl.maximum(radial2, 1.1754943508222875e-38)),
    )
    tl.store(Q + 3 * a, r * cosine, valid)
    tl.store(Q + 3 * a + 1, scale * c0, valid)
    tl.store(Q + 3 * a + 2, scale * c1, valid)
    north = -tl.div_rn(scale, r)
    tl.store(Jac + 6 * a, north * c0, valid)
    tl.store(Jac + 6 * a + 1, north * c1, valid)
    tl.store(Jac + 6 * a + 2, scale + curvature * c0 * c0, valid)
    tl.store(Jac + 6 * a + 3, curvature * c0 * c1, valid)
    tl.store(Jac + 6 * a + 4, curvature * c1 * c0, valid)
    tl.store(Jac + 6 * a + 5, scale + curvature * c1 * c1, valid)


@tr.jit
def decode_atoms(
    Source,
    Amp,
    Damp,
    Qi,
    Ji,
    Pi,
    Qo,
    Jo,
    Po,
    Ri,
    Ro,
    Shared,
    BoundsI,
    BoundsO,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    z0 = tl.load(Source + 6 * a, valid, other=0)
    z1 = tl.load(Source + 6 * a + 1, valid, other=0)
    square = z0 * z0 + z1 * z1
    safe = tl.maximum(square, 1.1754943508222875e-38)
    radius = libdevice.sqrt(safe)
    maximum = tl.load(Shared[0]).to(tl.float32)
    amp = tl.div_rn(maximum * z0, radius)
    alpha = tl.minimum(tl.maximum(tl.div_rn(square - 1.0, 3.0), 0.0), 1.0)
    tl.store(Amp + a, amp, valid)
    d0 = tl.where(square >= 1.1754943508222875e-38, tl.div_rn(-amp * z0, safe), 0.0)
    d1 = tl.where(square >= 1.1754943508222875e-38, tl.div_rn(-amp * z1, safe), 0.0)
    tl.store(Damp + 2 * a, d0 + tl.div_rn(maximum, radius), valid)
    tl.store(Damp + 2 * a + 1, d1, valid)
    tl.store(Pi + a, _precision(amp, alpha, Shared, BoundsI), valid)
    tl.store(Po + a, _precision(amp, alpha, Shared, BoundsO), valid)
    _geometry(Source, a, valid, Ri, Qi, Ji, 2)
    _geometry(Source, a, valid, Ro, Qo, Jo, 4)
