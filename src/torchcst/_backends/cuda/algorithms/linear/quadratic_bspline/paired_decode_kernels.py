"""Paired Meta stores with fixed atom ownership and unchanged FP64 decoding."""

import triton.experimental.gluon as tr
import triton.experimental.gluon.language as tl
from triton.experimental.gluon.language.extra import libdevice as lib


@tr.jit
def _pair(Meta, a, v0, v1, A: tl.constexpr, START: tl.constexpr, TILE: tl.constexpr):
    field = tl.arange(0, 2, layout=tl.SliceLayout(0, TILE))
    values = tl.where(field[None, :] == 0, v0[:, None], v1[:, None])
    tl.store(Meta + a[:, None] * 12 + START + field[None, :], values, a[:, None] < A)


@tr.jit
def decode(
    P,
    Meta,
    Origin,
    Spacing,
    Scalars,
    A: tl.constexpr,
    RATIO: tl.constexpr = False,
    WARPS: tl.constexpr = 1,
):
    tile: tl.constexpr = tl.BlockedLayout([1, 2], [32, 1], [WARPS, 1], [1, 0])
    a = tl.program_id(0) * 64 + tl.arange(0, 64, layout=tl.SliceLayout(1, tile))
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
    if RATIO:
        ratio = upper / lower
        valid = (a >= A) | ((lower > 0) & (ratio > 0) & (ratio < float("inf")))
        if tl.min(valid.to(tl.int32), 0) != 0:
            # Masked tail lanes cannot make a valid CTA take the slow branch.
            ratio = tl.where(a < A, ratio, 1.0)
            sigma = lower * lib.exp(alpha * lib.log(ratio))
        else:
            # Avoid ratio overflow for otherwise representable geometric means.
            sigma = lib.exp((1 - alpha) * lib.log(lower) + alpha * lib.log(upper))
    else:
        sigma = lib.exp((1 - alpha) * lib.log(lower) + alpha * lib.log(upper))
    sigma = tl.minimum(tl.maximum(sigma, lower), upper)
    radius = 2.449489742783178 * sigma
    co = tl.load(P + 4 * a + 2, a < A, 0).to(tl.float64)
    ci = tl.load(P + 4 * a + 3, a < A, 0).to(tl.float64)

    correction = tl.where(r2 >= 1.1754943508222875e-38, z0 * inv / safe, 0)
    _pair(Meta, a, amp, radius, A, 0, tile)
    _pair(Meta, a, co, ci, A, 2, tile)
    tl.store(Meta + a * 12 + 7, maximum_amp * (inv - z0 * correction), a < A)
    _pair(Meta, a, -maximum_amp * z1 * correction, 1.0 / radius, A, 8, tile)
    _pair(
        Meta,
        a,
        (a * 0).to(tl.float64) + 1.0 / tl.load(Spacing).to(tl.float64),
        (a * 0).to(tl.float64) + 1.0 / tl.load(Spacing + 1).to(tl.float64),
        A,
        10,
        tile,
    )
