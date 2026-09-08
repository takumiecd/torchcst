"""Fused exact factor contractions; CUDA-only optional Triton dependency."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _direction(
    DU,
    DV,
    X,
    A,
    B,
    ACTIVE,
    K: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    Q: tl.constexpr,
    CHECK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = True
    if CHECK:
        active = tl.load(ACTIVE)
    ai = n // I
    av = (
        tl.full((BLOCK,), 0, tl.float64)
        if X.dtype.element_ty == tl.float64
        else tl.full((BLOCK,), 0, tl.float32)
    )
    bv = av
    if active:
        for q in range(Q):
            av += tl.load(DU + n * Q + q, n < K * I, 0) * tl.load(
                X + ai * Q + q, ai < K, 0
            )
            ao = n // O
            bv += tl.load(DV + n * Q + q, n < K * O, 0) * tl.load(
                X + ao * Q + q, ao < K, 0
            )
    tl.store(A + n, av, n < K * I)
    tl.store(B + n, bv, n < K * O)


@tr.jit
def _gram(
    U,
    V,
    DU,
    DV,
    SU,
    SV,
    A,
    B,
    X,
    Y,
    ROW,
    COL,
    ACTIVE,
    K: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    Q: tl.constexpr,
    DAMP: tl.constexpr,
    WEIGHTED: tl.constexpr,
    CHECK: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
):
    t = tl.program_id(0) // Q
    q = tl.program_id(0) % Q
    active = True
    if CHECK:
        active = tl.load(ACTIVE)
    value = (
        tl.full((), 0, tl.float64)
        if U.dtype.element_ty == tl.float64
        else tl.full((), 0, tl.float32)
    )
    if active:
        for start in range(tl.cdiv(K, S)):
            s = start * S + tl.arange(0, S)
            uu = (
                tl.full((S,), 0, tl.float64)
                if U.dtype.element_ty == tl.float64
                else tl.full((S,), 0, tl.float32)
            )
            ua = uu
            duu = uu
            dua = uu
            vb = uu
            vv = uu
            dvb = uu
            dvv = uu
            for f in range(tl.cdiv(I, F)):
                i = f * F + tl.arange(0, F)
                us = tl.load(
                    SU + s[:, None] * I + i[None, :],
                    (s[:, None] < K) & (i[None, :] < I),
                    0,
                )
                a = tl.load(
                    A + s[:, None] * I + i[None, :],
                    (s[:, None] < K) & (i[None, :] < I),
                    0,
                )
                ut = tl.load(U + t * I + i, i < I, 0)
                dut = tl.load(DU + (t * I + i) * Q + q, i < I, 0)
                if WEIGHTED:
                    w = tl.load(COL + i, i < I, 0)
                    ut *= w
                    dut *= w
                uu += tl.sum(us * ut[None, :], 1)
                ua += tl.sum(a * ut[None, :], 1)
                duu += tl.sum(us * dut[None, :], 1)
                dua += tl.sum(a * dut[None, :], 1)
            for f in range(tl.cdiv(O, F)):
                o = f * F + tl.arange(0, F)
                vs = tl.load(
                    SV + s[:, None] * O + o[None, :],
                    (s[:, None] < K) & (o[None, :] < O),
                    0,
                )
                b = tl.load(
                    B + s[:, None] * O + o[None, :],
                    (s[:, None] < K) & (o[None, :] < O),
                    0,
                )
                vt = tl.load(V + t * O + o, o < O, 0)
                dvt = tl.load(DV + (t * O + o) * Q + q, o < O, 0)
                if WEIGHTED:
                    w = tl.load(ROW + o, o < O, 0)
                    vt *= w
                    dvt *= w
                vb += tl.sum(b * vt[None, :], 1)
                vv += tl.sum(vs * vt[None, :], 1)
                dvb += tl.sum(b * dvt[None, :], 1)
                dvv += tl.sum(vs * dvt[None, :], 1)
            value += tl.sum(uu * dvb + ua * dvv + duu * vb + dua * vv, 0)
        value += DAMP * tl.load(X + t * Q + q)
    tl.store(Y + t * Q + q, value)


def cross(target, source, x, *, row=None, column=None, damping=0.0, active=None):
    u, v, du, dv = target._u, target._v, target._du, target._dv
    su, sv, sdu, sdv = source._u, source._v, source._du, source._dv
    k, i = u.shape
    o = v.shape[1]
    q = x.shape[1]
    if not x.is_cuda or any(
        not t.is_contiguous() for t in (u, v, du, dv, su, sv, sdu, sdv, x)
    ):
        raise ValueError(
            "Triton tangent actions require contiguous CUDA factors/vectors"
        )
    a, b = torch.empty_like(su), torch.empty_like(sv)
    result = torch.empty_like(x)
    _direction[(tr.cdiv(k * max(i, o), 256),)](
        sdu,
        sdv,
        x,
        a,
        b,
        active,
        K=k,
        I=i,
        O=o,
        Q=q,
        CHECK=active is not None,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    _gram[(k * q,)](
        u,
        v,
        du,
        dv,
        su,
        sv,
        a,
        b,
        x,
        result,
        row,
        column,
        active,
        K=k,
        I=i,
        O=o,
        Q=q,
        DAMP=damping,
        WEIGHTED=row is not None,
        CHECK=active is not None,
        S=8,
        F=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@tr.jit
def _secular(
    V, C, X, SHIFT, N: tl.constexpr, RADIUS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    v = tl.load(V + i, i < N, 0)
    c = tl.load(C + i, i < N, 0)
    radius = tl.full((), RADIUS, tl.float64)
    cutoff = 8 * 2.220446049250313e-16 * tl.maximum(tl.max(v, 0), 1e-30)
    scale = tl.max(tl.abs(c), 0)
    normalized = c / tl.where(scale > 0, scale, 1)
    norm_rhs = scale * tl.sqrt(tl.sum(normalized * normalized, 0))
    visible = v > cutoff
    null_ok = (
        tl.sum(
            (
                ~visible
                & (tl.abs(c) > 8 * 2.220446049250313e-16 * tl.maximum(norm_rhs, 1e-30))
            ).to(tl.int32),
            0,
        )
        == 0
    )
    x = tl.where(visible, c / tl.where(visible, v, 1), 0)
    scale_x = tl.max(tl.abs(x), 0)
    normalized_x = x / tl.where(scale_x > 0, scale_x, 1)
    norm_x = scale_x * tl.sqrt(tl.sum(normalized_x * normalized_x, 0))
    shift = tl.full((), 0, tl.float64)
    if ~null_ok or norm_x > radius:
        low = tl.full((), 0, tl.float64)
        high = norm_rhs / radius
        for _ in range(80):
            mid = (low + high) * 0.5
            y = c / tl.maximum(v + mid, 1e-300)
            scale_y = tl.max(tl.abs(y), 0)
            normalized_y = y / tl.where(scale_y > 0, scale_y, 1)
            norm_y = scale_y * tl.sqrt(tl.sum(normalized_y * normalized_y, 0))
            low = tl.where(norm_y > radius, mid, low)
            high = tl.where(norm_y > radius, high, mid)
        shift = high
        x = c / tl.maximum(v + high, 1e-300)
    tl.store(X + i, x, i < N)
    tl.store(SHIFT, shift)


def spectral_solution(values, coeff, radius):
    out = torch.empty_like(coeff)
    shift = torch.empty((), device=values.device, dtype=values.dtype)
    _secular[(1,)](
        values,
        coeff,
        out,
        shift,
        N=values.numel(),
        RADIUS=radius,
        BLOCK=tr.next_power_of_2(values.numel()),
        enable_fp_fusion=False,
    )
    return out, shift


@tr.jit
def _metric_force(
    U,
    V,
    A,
    B,
    ROW,
    COL,
    FORCE,
    ACTIVE,
    K: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    EPS: tl.constexpr,
    RATE: tl.constexpr,
    CHECK: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
):
    i = tl.program_id(0) * F + tl.arange(0, F)
    r = tl.program_id(1)
    active = True
    if CHECK:
        active = tl.load(ACTIVE)
    result = tl.full((F,), 0, U.dtype.element_ty)
    if active:
        for start in range(tl.cdiv(K, S)):
            k = start * S + tl.arange(0, S)
            u = tl.load(
                U + k[:, None] * I + i[None, :], (k[:, None] < K) & (i[None, :] < I), 0
            )
            a = tl.load(
                A + k[:, None] * I + i[None, :], (k[:, None] < K) & (i[None, :] < I), 0
            )
            v = tl.load(V + k * O + START + r, k < K, 0)
            b = tl.load(B + k * O + START + r, k < K, 0)
            result += tl.sum(u * b[:, None] + a * v[:, None], 0)
        weight = tl.load(ROW + START + r) * tl.load(COL + i, i < I, 0)
        result *= (weight + tl.full((), EPS, U.dtype.element_ty)) / tl.full(
            (), RATE, U.dtype.element_ty
        )
    tl.store(FORCE + r * I + i, result, i < I)


@tr.jit
def _metric_pullback(
    U,
    V,
    DU,
    DV,
    FORCE,
    Y,
    ACTIVE,
    I: tl.constexpr,
    O: tl.constexpr,
    Q: tl.constexpr,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    CHECK: tl.constexpr,
    R: tl.constexpr,
    F: tl.constexpr,
):
    k = tl.program_id(0) // Q
    q = tl.program_id(0) % Q
    r = tl.arange(0, R)
    active = True
    if CHECK:
        active = tl.load(ACTIVE)
    result = tl.full((), 0, U.dtype.element_ty)
    if active:
        v = tl.load(V + k * O + START + r, r < ROWS, 0)
        dv = tl.load(DV + (k * O + START + r) * Q + q, r < ROWS, 0)
        for start in range(tl.cdiv(I, F)):
            i = start * F + tl.arange(0, F)
            force = tl.load(
                FORCE + r[:, None] * I + i[None, :],
                (r[:, None] < ROWS) & (i[None, :] < I),
                0,
            )
            u = tl.load(U + k * I + i, i < I, 0)
            du = tl.load(DU + (k * I + i) * Q + q, i < I, 0)
            result += tl.sum(
                tl.sum(
                    force * (v[:, None] * du[None, :] + dv[:, None] * u[None, :]), 1
                ),
                0,
            )
    if START > 0:
        result += tl.load(Y + k * Q + q)
    tl.store(Y + k * Q + q, result)


def metric_action(prepared, x, row, column, eps, rate, active=None, *, source=None):
    """Stream bounded visible row tiles: JVP, diagonal metric, then VJP.

    Arithmetic is O(K*q*I*O), avoiding atom-pair contractions for each Krylov
    iteration. Scratch is two factor directions plus at most 16 visible rows.
    """
    p = prepared
    source = p if source is None else source
    k, i = p._u.shape
    o, q = p._v.shape[1], x.shape[1]
    a, b, result = torch.empty_like(p._u), torch.empty_like(p._v), torch.empty_like(x)
    _direction[(tr.cdiv(k * max(i, o), 256),)](
        source._du,
        source._dv,
        x,
        a,
        b,
        active,
        K=k,
        I=i,
        O=o,
        Q=q,
        CHECK=active is not None,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    for start in range(0, o, 16):
        rows = min(16, o - start)
        force = torch.empty((rows, i), device=x.device, dtype=x.dtype)
        _metric_force[(tr.cdiv(i, 64), rows)](
            source._u,
            source._v,
            a,
            b,
            row,
            column,
            force,
            active,
            K=k,
            I=i,
            O=o,
            START=start,
            ROWS=rows,
            EPS=eps,
            RATE=rate,
            CHECK=active is not None,
            S=16,
            F=64,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _metric_pullback[(k * q,)](
            p._u,
            p._v,
            p._du,
            p._dv,
            force,
            result,
            active,
            I=i,
            O=o,
            Q=q,
            START=start,
            ROWS=rows,
            CHECK=active is not None,
            R=tr.next_power_of_2(rows),
            F=128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result
