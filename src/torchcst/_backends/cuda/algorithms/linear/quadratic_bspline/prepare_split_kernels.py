"""Separate Polar decoding and unchanged sampled normalization to lower register use."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

from .kernels import _norm_pair


@tr.jit
def decode(P, Meta, Origin, Spacing, Scalars, A: tl.constexpr):
    a = tl.program_id(0) * 64 + tl.arange(0, 64)
    z0 = tl.load(P + 4 * a, a < A, 0).to(tl.float64)
    z1 = tl.load(P + 4 * a + 1, a < A, 0).to(tl.float64)
    r2 = z0 * z0 + z1 * z1
    safe = tl.maximum(r2, 1.1754943508222875e-38)
    inv = 1 / lib.sqrt(safe)
    maximum_amp = tl.load(Scalars[0]).to(tl.float64)
    amp = maximum_amp * z0 * inv
    alpha = tl.minimum(tl.maximum((r2 - 1) / 3, 0), 1)
    w = amp / tl.load(Scalars[1]).to(tl.float64)
    x = w * w
    minimum = tl.load(Scalars[5]).to(tl.float64)
    birth = tl.load(Scalars[6]).to(tl.float64)
    maximum = tl.load(Scalars[7]).to(tl.float64)
    kappa = tl.load(Scalars[2]).to(tl.float64)
    power = tl.load(Scalars[4]).to(tl.float64)
    if power == 1.0:
        ux = x
    else:
        ux = lib.pow(x, power)
    upper = minimum + (maximum - minimum) * kappa / (kappa + ux)
    upper = tl.maximum(upper, tl.load(Scalars[8]).to(tl.float64))
    lower = minimum + (birth - minimum) / (1 + tl.load(Scalars[3]).to(tl.float64) * x)
    upper = tl.maximum(upper, lower)
    sigma = lib.exp((1 - alpha) * lib.log(lower) + alpha * lib.log(upper))
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    radius = 2.449489742783178 * sigma
    co = tl.load(P + 4 * a + 2, a < A, 0).to(tl.float64)
    ci = tl.load(P + 4 * a + 3, a < A, 0).to(tl.float64)

    correction = tl.where(r2 >= 1.1754943508222875e-38, z0 * inv / safe, 0)
    tl.store(Meta + a * 12, amp, a < A)
    tl.store(Meta + a * 12 + 1, radius, a < A)
    tl.store(Meta + a * 12 + 2, co, a < A)
    tl.store(Meta + a * 12 + 3, ci, a < A)
    tl.store(Meta + a * 12 + 7, maximum_amp * (inv - z0 * correction), a < A)
    tl.store(Meta + a * 12 + 8, -maximum_amp * z1 * correction, a < A)
    tl.store(Meta + a * 12 + 9, 1.0 / radius, a < A)
    tl.store(Meta + a * 12 + 10, 1.0 / tl.load(Spacing).to(tl.float64), a < A)
    tl.store(Meta + a * 12 + 11, 1.0 / tl.load(Spacing + 1).to(tl.float64), a < A)


@tr.jit
def normalize(
    Meta,
    Origin,
    Spacing,
    Moments,
    A: tl.constexpr,
    NO: tl.constexpr,
    NI: tl.constexpr,
    ML: tl.constexpr,
    FLOOR: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    LOOKUP: tl.constexpr,
    FMA: tl.constexpr,
    CORE32: tl.constexpr,
    HOIST: tl.constexpr = False,
):
    a = tl.program_id(0) * 64 + tl.arange(0, 64)
    radius = tl.load(Meta + 12 * a + 1, a < A, 1)
    co = tl.load(Meta + 12 * a + 2, a < A, 0)
    ci = tl.load(Meta + 12 * a + 3, a < A, 0)
    qo, vo, qi, vi = _norm_pair(
        co,
        ci,
        radius,
        Origin,
        Spacing,
        Moments,
        NO,
        NI,
        ML,
        RECIPROCAL,
        LOOKUP,
        FLOOR,
        FMA,
        CORE32,
        HOIST,
    )
    denominator = lib.sqrt(qo * qi)
    tl.store(Meta + a * 12 + 4, 1 / tl.maximum(denominator, FLOOR), a < A)
    tl.store(
        Meta + a * 12 + 5,
        tl.where(denominator >= FLOOR, vo / tl.where(qo > 0, qo, 1), 0),
        a < A,
    )
    tl.store(
        Meta + a * 12 + 6,
        tl.where(denominator >= FLOOR, vi / tl.where(qi > 0, qi, 1), 0),
        a < A,
    )
