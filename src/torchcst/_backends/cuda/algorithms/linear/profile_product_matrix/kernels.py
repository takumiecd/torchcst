"""Complete support patches, FP32 aggregate weights, full norm derivatives."""

import triton as tr
import triton.language as tl

from ..local_product.kernels import _raw, _store_param_cotangent
from ..profile_product_global.grouped_kernels import _positions


@tr.jit
def ieee_matmul(
    L,
    R,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        left = tl.load(
            L + m[:, None] * L0 + kk[None, :] * L1,
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        right = tl.load(
            R + kk[:, None] * R0 + n[None, :] * R1,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        acc = tl.dot(left, right, acc, input_precision="ieee")
    tl.store(
        Out + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N)
    )


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
):
    a = tl.program_id(0)
    amp, inv = tl.load(P + a), tl.load(P + A + a)
    ci, co = tl.load(P + 2 * A + a), tl.load(P + 3 * A + a)
    sv, su = tl.load(P + 4 * A + a), tl.load(P + 5 * A + a)
    ilo, ihi = tl.load(P + 9 * A + a).to(tl.int32), tl.load(P + 10 * A + a).to(tl.int32)
    olo, ohi = (
        tl.load(P + 11 * A + a).to(tl.int32),
        tl.load(P + 12 * A + a).to(tl.int32),
    )
    k = tl.arange(0, PATCH)
    for istart in range(0, tl.maximum(ohi - olo, 0), PATCH):
        i = olo + istart + k
        u, _ = _raw(OO + i * S - co, inv)
        u = tl.div_rn(u, su)
        for jstart in range(0, tl.maximum(ihi - ilo, 0), PATCH):
            j = ilo + jstart + k
            v, _ = _raw(_positions(j, Pitch, OI, S, TILE) - ci, inv)
            v = tl.div_rn(v, sv)
            mask = (
                (i[:, None] < ohi)
                & (i[:, None] < NO)
                & (j[None, :] < ihi)
                & (j[None, :] < NI)
            )
            tl.atomic_add(
                W + i[:, None] * NI + j[None, :],
                amp * u[:, None] * v[None, :],
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
):
    a = tl.program_id(0)
    amp, inv = tl.load(P + a), tl.load(P + A + a)
    ci, co = tl.load(P + 2 * A + a), tl.load(P + 3 * A + a)
    sv, su = tl.load(P + 4 * A + a), tl.load(P + 5 * A + a)
    gv, gu = tl.load(P + 6 * A + a), tl.load(P + 7 * A + a)
    flags = tl.load(P + 8 * A + a).to(tl.int32)
    ilo, ihi = tl.load(P + 9 * A + a).to(tl.int32), tl.load(P + 10 * A + a).to(tl.int32)
    olo, ohi = (
        tl.load(P + 11 * A + a).to(tl.int32),
        tl.load(P + 12 * A + a).to(tl.int32),
    )
    k = tl.arange(0, PATCH)
    da, dco, dci = (
        tl.full((PATCH, PATCH), 0, tl.float32),
        tl.full((PATCH, PATCH), 0, tl.float32),
        tl.full((PATCH, PATCH), 0, tl.float32),
    )
    for istart in range(0, tl.maximum(ohi - olo, 0), PATCH):
        i = olo + istart + k
        u, du = _raw(OO + i * S - co, inv)
        u = tl.div_rn(u, su)
        du = tl.where((flags & 2) != 0, 0.0, tl.div_rn(du, su) - gu * u)
        for jstart in range(0, tl.maximum(ihi - ilo, 0), PATCH):
            j = ilo + jstart + k
            v, dv = _raw(_positions(j, Pitch, OI, S, TILE) - ci, inv)
            v = tl.div_rn(v, sv)
            dv = tl.where((flags & 1) != 0, 0.0, tl.div_rn(dv, sv) - gv * v)
            mask = (
                (i[:, None] < ohi)
                & (i[:, None] < NO)
                & (j[None, :] < ihi)
                & (j[None, :] < NI)
            )
            dw = tl.load(DW + i[:, None] * D0 + j[None, :] * D1, mask, 0)
            da += dw * u[:, None] * v[None, :]
            dco += dw * du[:, None] * v[None, :]
            dci += dw * u[:, None] * dv[None, :]
    _store_param_cotangent(
        DP,
        Source,
        AmplitudeMax,
        a,
        a < A,
        tl.sum(tl.sum(da, 1), 0),
        amp * tl.sum(tl.sum(dci, 1), 0),
        amp * tl.sum(tl.sum(dco, 1), 0),
        True,
        CENTER_OUTPUT_FIRST=True,
    )
