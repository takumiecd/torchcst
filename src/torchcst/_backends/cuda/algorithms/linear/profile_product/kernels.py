"""Global product norm and complete center derivatives in local metadata."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _polar_atom, _raw


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
    BOUNDS: tl.constexpr = False,
):
    a = tl.program_id(0)
    amp, inv = _polar_atom(Source, a, Scalars)
    # Single Chart's canonical order is output center, input center.
    co, ci = tl.load(Source + 4 * a + 2), tl.load(Source + 4 * a + 3)
    j, i = tl.arange(0, PK), tl.arange(0, PN)
    v, dv = _raw(OI + j * S - ci, inv)
    u, du = _raw(OO + i * S - co, inv)
    v, dv = tl.where(j < KI, v, 0.0), tl.where(j < KI, dv, 0.0)
    u, du = tl.where(i < NO, u, 0.0), tl.where(i < NO, du, 0.0)
    nv2, nu2 = tl.sum(v * v, 0), tl.sum(u * u, 0)
    nv, nu = tl.sqrt(nv2), tl.sqrt(nu2)
    active = nv * nu >= FLOOR
    # Axis-normalized factors when active; split the single floor otherwise.
    # Only their product defines W. gamma is d(log D)/d(center), so the
    # existing contraction VJP includes the complete denominator derivative.
    sv, su = tl.where(active, nv, tl.sqrt(FLOOR)), tl.where(active, nu, tl.sqrt(FLOOR))
    gv = tl.where(
        active,
        tl.div_rn(tl.sum(v * dv, 0), tl.maximum(nv2, 1.1754943508222875e-38)),
        0.0,
    )
    gu = tl.where(
        active,
        tl.div_rn(tl.sum(u * du, 0), tl.maximum(nu2, 1.1754943508222875e-38)),
        0.0,
    )
    tl.store(P + a, amp)
    tl.store(P + A + a, inv)
    tl.store(P + 2 * A + a, ci)
    tl.store(P + 3 * A + a, co)
    tl.store(P + 4 * A + a, sv)
    tl.store(P + 5 * A + a, su)
    tl.store(P + 6 * A + a, gv)
    tl.store(P + 7 * A + a, gu)
    flags = (active & (tl.sum((v > 0).to(tl.int32), 0) == 1)).to(tl.int32)
    flags |= (active & (tl.sum((u > 0).to(tl.int32), 0) == 1)).to(tl.int32) << 1
    tl.store(P + 8 * A + a, flags.to(tl.float32))
    if BOUNDS:
        tl.store(P + 9 * A + a, tl.min(tl.where(v > 0, j, KI), 0).to(tl.float32))
        tl.store(P + 10 * A + a, tl.max(tl.where(v > 0, j + 1, 0), 0).to(tl.float32))
        tl.store(P + 11 * A + a, tl.min(tl.where(u > 0, i, NO), 0).to(tl.float32))
        tl.store(P + 12 * A + a, tl.max(tl.where(u > 0, i + 1, 0), 0).to(tl.float32))
