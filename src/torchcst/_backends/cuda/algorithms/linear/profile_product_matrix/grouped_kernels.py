"""Group canonical atoms while visiting every current support patch."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent
from ..profile_product_global.grouped_kernels import _positions


@tr.jit
def assemble(
    P,
    W,
    Pitch,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
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
        u, _ = _raw(OO + i * S - co[:, None], inv[:, None])
        u = tl.where(
            valid[:, None] & (i < ohi[:, None]) & (i < NO),
            tl.div_rn(u, su[:, None]),
            0.0,
        )
        for jstart in range(0, tl.max(tl.maximum(ihi - ilo, 0), 0), PATCH):
            j = ilo[:, None] + jstart + k[None, :]
            v, _ = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
            v = tl.where(
                valid[:, None] & (j < ihi[:, None]) & (j < NI),
                tl.div_rn(v, sv[:, None]),
                0.0,
            )
            mask = (
                valid[:, None, None]
                & (i[:, :, None] < ohi[:, None, None])
                & (i[:, :, None] < NO)
                & (j[:, None, :] < ihi[:, None, None])
                & (j[:, None, :] < NI)
            )
            tl.atomic_add(
                W + i[:, :, None] * NI + j[:, None, :],
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
    Pitch,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    TILE: tl.constexpr,
    S: tl.constexpr,
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
    da, dco, dci = (
        tl.full((GROUP, PATCH, PATCH), 0, tl.float32),
        tl.full((GROUP, PATCH, PATCH), 0, tl.float32),
        tl.full((GROUP, PATCH, PATCH), 0, tl.float32),
    )
    for istart in range(0, tl.max(tl.maximum(ohi - olo, 0), 0), PATCH):
        i = olo[:, None] + istart + k[None, :]
        u, du = _raw(OO + i * S - co[:, None], inv[:, None])
        u = tl.div_rn(u, su[:, None])
        du = tl.where(
            (flags[:, None] & 2) != 0, 0.0, tl.div_rn(du, su[:, None]) - gu[:, None] * u
        )
        live_i = valid[:, None] & (i < ohi[:, None]) & (i < NO)
        u, du = tl.where(live_i, u, 0.0), tl.where(live_i, du, 0.0)
        for jstart in range(0, tl.max(tl.maximum(ihi - ilo, 0), 0), PATCH):
            j = ilo[:, None] + jstart + k[None, :]
            v, dv = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
            v = tl.div_rn(v, sv[:, None])
            dv = tl.where(
                (flags[:, None] & 1) != 0,
                0.0,
                tl.div_rn(dv, sv[:, None]) - gv[:, None] * v,
            )
            live_j = valid[:, None] & (j < ihi[:, None]) & (j < NI)
            v, dv = tl.where(live_j, v, 0.0), tl.where(live_j, dv, 0.0)
            mask = (
                valid[:, None, None]
                & (i[:, :, None] < ohi[:, None, None])
                & (i[:, :, None] < NO)
                & (j[:, None, :] < ihi[:, None, None])
                & (j[:, None, :] < NI)
            )
            dw = tl.load(DW + i[:, :, None] * D0 + j[:, None, :] * D1, mask, 0)
            da += dw * u[:, :, None] * v[:, None, :]
            dco += dw * du[:, :, None] * v[:, None, :]
            dci += dw * u[:, :, None] * dv[:, None, :]
    _store_param_cotangent(
        DP,
        Source,
        AmplitudeMax,
        a,
        a < A,
        tl.sum(tl.sum(da, 2), 1),
        amp * tl.sum(tl.sum(dci, 2), 1),
        amp * tl.sum(tl.sum(dco, 2), 1),
        True,
        CENTER_OUTPUT_FIRST=True,
    )
