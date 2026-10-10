"""Canonical Product VJP with H and its input-center derivative fused.

The H argument preserves the unfused launch signature; this kernel never reads it.
All normalization, support and singleton decisions come from the forward snapshot.
"""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent
from ..profile_product_global.grouped_kernels import _positions


@tr.jit
def backward_atoms(
    X,
    DY,
    Views,
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
    p = Views
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
        # Singleton input factors still contribute to H and the amplitude/output
        # VJP. Only their input-center derivative is identically zero.
        hi = tl.where(ohi > olo, ihi, ilo)
        hacc, hci = (
            tl.full((GROUP, BM, BK), 0.0, tl.float32),
            tl.full((GROUP, BM, BK), 0.0, tl.float32),
        )
        for start in range(0, tl.max(tl.maximum(hi - ilo, 0), 0), BK):
            j = ilo[:, None] + start + k[None, :]
            v, dv = _raw(_positions(j, Pitch, OI, S, TILE) - ci[:, None], inv[:, None])
            v = tl.div_rn(v, sv[:, None])
            dv = tl.where(
                (flags[:, None] & 1) != 0,
                0.0,
                tl.div_rn(dv, sv[:, None]) - gv[:, None] * v,
            )
            mask = (
                valid[:, None, None]
                & (rows[None, :, None] < B)
                & (j[:, None, :] < hi[:, None, None])
                & (j[:, None, :] < NI)
            )
            x = tl.load(X + rows[None, :, None] * X0 + j[:, None, :] * X1, mask, 0)
            hacc += x * v[:, None, :]
            hci += x * dv[:, None, :]
        h = tl.sum(hacc, 2)
        da, dco = tl.sum(h * g, 1), amp * tl.sum(h * tl.sum(dc, 2), 1)
        dci = amp * tl.sum(tl.sum(hci, 2) * g, 1)
        _store_param_cotangent(
            DP,
            Source,
            AmplitudeMax,
            a,
            valid,
            da,
            dci,
            dco,
            True,
            CENTER_OUTPUT_FIRST=True,
        )
