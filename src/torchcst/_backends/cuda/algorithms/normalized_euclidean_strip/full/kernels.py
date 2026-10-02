"""Validated whole-weight CUDA implementation of globally normalized Triweight."""

import triton as tr
import triton.language as tl

COMPILED_KERNELS = {}
_ATLAS_CACHE = {}


def atlas_tables(device):
    key = str(device)
    if key not in _ATLAS_CACHE:
        from torchcst._backends.cuda.algorithms.normalized_euclidean_strip._shared.atlas_table import (
            build_atlas,
        )

        codes, counts = build_atlas()
        _ATLAS_CACHE[key] = (
            codes.to(device),
            counts.to(device),
        )
    return _ATLAS_CACHE[key]


@tr.jit
def _sum_norm_count(n0, c0, n1, c1):
    return n0 + n1, c0 + c1


@tr.jit
def _sum_four(a0, b0, c0, d0, a1, b1, c1, d1):
    return a0 + a1, b0 + b1, c0 + c1, d0 + d1


@tr.jit
def _point_math(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    original_a,
    amp,
    precision,
    c0,
    c1,
    c2,
    r0,
    r1,
    r2,
    valid,
    H: tl.constexpr,
    J: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    BACK: tl.constexpr,
    SAVED_FLAGS: tl.constexpr,
    TUPLE_GRADS: tl.constexpr,
):
    d0 = O0 + r0 * S0 - c0
    d1 = O1 + r1 * S1 - c1
    d2 = O2 + r2 * S2 - c2
    q = (d0 * d0 + d1 * d1 + d2 * d2) * precision
    t = tl.where(valid, tl.maximum(1.0 - q, 0.0), 0.0)
    k = t * t * t
    address = r0 * (H * J) + r1 * J + r2
    if not BACK:
        if SAVED_FLAGS:
            norm2, positive = tl.reduce(
                (k * k, (k > 0).to(tl.int32)), 0, _sum_norm_count
            )
        else:
            norm2 = tl.sum(k * k, 0)
        raw_norm = tl.sqrt_rn(norm2)
        norm = tl.maximum(raw_norm, 1.0e-6)
        tl.store(Norm + original_a, norm)
        if SAVED_FLAGS:
            live = raw_norm >= 1.0e-6
            singleton = (positive == 1) & live
            flags = live.to(tl.int32) | (singleton.to(tl.int32) << 1)
            tl.store(SupportFlags + original_a, flags)
        effective_amp = amp / norm
        tl.atomic_add(W + address, effective_amp * k, valid & (k > 0), sem="relaxed")
    else:
        dw = tl.load(W + address, valid & (k > 0), 0.0)
        norm = tl.load(Norm + original_a)
        effective_amp = amp / norm
        g0 = tl.sum(dw * k, 0)
        if SAVED_FLAGS:
            flags = tl.load(SupportFlags + original_a).to(tl.int32)
            live_norm = (flags & 1) != 0
            singleton = (flags & 2) != 0
        else:
            norm2 = tl.sum(k * k, 0)
            # Match the actual RN forward root and clamp-equality derivative.
            live_norm = tl.sqrt_rn(norm2) >= 1.0e-6
            singleton = (tl.sum((k > 0).to(tl.int32), 0) == 1) & live_norm
        gauge = tl.where(live_norm, g0 / (norm * norm), 0.0)
        residual = dw - gauge * k
        dc = tl.where(
            singleton, 0.0, effective_amp * 6.0 * precision * t * t * residual
        )
        gl = tl.where(
            singleton | (k == 0), 0.0, effective_amp * 6.0 * q * t * t * residual
        )
        if TUPLE_GRADS:
            gl_sum, gc0, gc1, gc2 = tl.reduce(
                (gl, dc * d0, dc * d1, dc * d2), 0, _sum_four
            )
        else:
            gl_sum = tl.sum(gl, 0)
            gc0 = tl.sum(dc * d0, 0)
            gc1 = tl.sum(dc * d1, 0)
            gc2 = tl.sum(dc * d2, 0)
        log = tl.load(P + 5 * original_a + 1)
        tl.store(DP + 5 * original_a, g0 / norm)
        tl.store(
            DP + 5 * original_a + 1, tl.where((log >= LO) & (log <= HI), gl_sum, 0.0)
        )
        tl.store(DP + 5 * original_a + 2, gc0)
        tl.store(DP + 5 * original_a + 3, gc1)
        tl.store(DP + 5 * original_a + 4, gc2)


@tr.jit
def _wide_math(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Offsets,
    original_a,
    amp,
    precision,
    c0,
    c1,
    c2,
    N: tl.constexpr,
    H: tl.constexpr,
    J: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    COUNT: tl.constexpr,
    BACK: tl.constexpr,
    SAVED_FLAGS: tl.constexpr,
):
    lane = tl.arange(0, 512)
    near0 = tl.floor((c0 - O0) / S0 + 0.5).to(tl.int32)
    near1 = tl.floor((c1 - O1) / S1 + 0.5).to(tl.int32)
    near2 = tl.floor((c2 - O2) / S2 + 0.5).to(tl.int32)
    norm2 = 0.0
    positive = 0
    g0 = 0.0
    for chunk in range(2):
        idx = lane + chunk * 512
        code = tl.load(Offsets + idx, idx < COUNT, 0)
        r0 = near0 + (code & 15) - 4
        r1 = near1 + ((code >> 4) & 31) - 8
        r2 = near2 + ((code >> 9) & 31) - 8
        valid = (
            (idx < COUNT)
            & (r0 >= 0)
            & (r0 < N)
            & (r1 >= 0)
            & (r1 < H)
            & (r2 >= 0)
            & (r2 < J)
        )
        d0 = O0 + r0 * S0 - c0
        d1 = O1 + r1 * S1 - c1
        d2 = O2 + r2 * S2 - c2
        q = (d0 * d0 + d1 * d1 + d2 * d2) * precision
        t = tl.where(valid, tl.maximum(1.0 - q, 0.0), 0.0)
        k = t * t * t
        if not BACK or not SAVED_FLAGS:
            norm2 += tl.sum(k * k, 0)
            positive += tl.sum((k > 0).to(tl.int32), 0)
        if BACK:
            address = r0 * (H * J) + r1 * J + r2
            dw = tl.load(W + address, valid & (k > 0), 0.0)
            g0 += tl.sum(dw * k, 0)
    if BACK:
        norm = tl.load(Norm + original_a)
        if SAVED_FLAGS:
            flags = tl.load(SupportFlags + original_a).to(tl.int32)
            live = (flags & 1) != 0
            singleton = (flags & 2) != 0
        else:
            live = tl.sqrt_rn(norm2) >= 1.0e-6
            singleton = (positive == 1) & live
    else:
        raw_norm = tl.sqrt_rn(norm2)
        norm = tl.maximum(raw_norm, 1.0e-6)
        live = raw_norm >= 1.0e-6
        singleton = (positive == 1) & live
        tl.store(Norm + original_a, norm)
        if SAVED_FLAGS:
            tl.store(
                SupportFlags + original_a,
                live.to(tl.int32) | (singleton.to(tl.int32) << 1),
            )
    effective_amp = amp / norm
    gauge = tl.where(live, g0 / (norm * norm), 0.0)
    gl_sum = 0.0
    gc0 = 0.0
    gc1 = 0.0
    gc2 = 0.0
    for chunk in range(2):
        idx = lane + chunk * 512
        code = tl.load(Offsets + idx, idx < COUNT, 0)
        r0 = near0 + (code & 15) - 4
        r1 = near1 + ((code >> 4) & 31) - 8
        r2 = near2 + ((code >> 9) & 31) - 8
        valid = (
            (idx < COUNT)
            & (r0 >= 0)
            & (r0 < N)
            & (r1 >= 0)
            & (r1 < H)
            & (r2 >= 0)
            & (r2 < J)
        )
        d0 = O0 + r0 * S0 - c0
        d1 = O1 + r1 * S1 - c1
        d2 = O2 + r2 * S2 - c2
        q = (d0 * d0 + d1 * d1 + d2 * d2) * precision
        t = tl.where(valid, tl.maximum(1.0 - q, 0.0), 0.0)
        k = t * t * t
        address = r0 * (H * J) + r1 * J + r2
        if not BACK:
            tl.atomic_add(
                W + address, effective_amp * k, valid & (k > 0), sem="relaxed"
            )
        else:
            dw = tl.load(W + address, valid & (k > 0), 0.0)
            residual = dw - gauge * k
            dc = tl.where(
                singleton, 0.0, effective_amp * 6.0 * precision * t * t * residual
            )
            gl = tl.where(
                singleton | (k == 0), 0.0, effective_amp * 6.0 * q * t * t * residual
            )
            gl_sum += tl.sum(gl, 0)
            gc0 += tl.sum(dc * d0, 0)
            gc1 += tl.sum(dc * d1, 0)
            gc2 += tl.sum(dc * d2, 0)
    if BACK:
        log = tl.load(P + original_a * 5 + 1)
        tl.store(DP + original_a * 5, g0 / norm)
        tl.store(
            DP + original_a * 5 + 1, tl.where((log >= LO) & (log <= HI), gl_sum, 0.0)
        )
        tl.store(DP + original_a * 5 + 2, gc0)
        tl.store(DP + original_a * 5 + 3, gc1)
        tl.store(DP + original_a * 5 + 4, gc2)


@tr.jit
def _site_q(
    r,
    h,
    j,
    c0,
    c1,
    c2,
    precision,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    d0 = O0 + r * S0 - c0
    d1 = O1 + h * S1 - c1
    d2 = O2 + j * S2 - c2
    return (d0 * d0 + d1 * d1 + d2 * d2) * precision


@tr.jit
def _routed_atoms(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Order,
    Offsets,
    Atlas,
    AtlasCounts,
    N: tl.constexpr,
    H: tl.constexpr,
    J: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B: tl.constexpr,
    BACK: tl.constexpr,
    SORTED: tl.constexpr,
    BALL: tl.constexpr,
    COUNT: tl.constexpr,
    SAVED_FLAGS: tl.constexpr,
    TUPLE_GRADS: tl.constexpr,
    ATLAS_GEOMETRY: tl.constexpr,
):
    a = tl.program_id(0)
    original_a = tl.load(Order + a) if SORTED else a
    amp = tl.load(P + 5 * original_a)
    log = tl.load(P + 5 * original_a + 1)
    precision = tl.exp(-2.0 * tl.minimum(tl.maximum(log, LO), HI))
    c0 = tl.load(P + 5 * original_a + 2)
    c1 = tl.load(P + 5 * original_a + 3)
    c2 = tl.load(P + 5 * original_a + 4)
    u0 = (c0 - O0) / S0
    u1 = (c1 - O1) / S1
    u2 = (c2 - O2) / S2
    near0 = tl.floor(u0 + 0.5).to(tl.int32)
    near1 = tl.floor(u1 + 0.5).to(tl.int32)
    near2 = tl.floor(u2 + 0.5).to(tl.int32)
    phase0 = tl.minimum(
        tl.maximum(tl.floor((u0 - near0 + 0.5) * 16).to(tl.int32), 0), 15
    )
    phase1 = tl.minimum(
        tl.maximum(tl.floor((u1 - near1 + 0.5) * 16).to(tl.int32), 0), 15
    )
    phase2 = tl.minimum(
        tl.maximum(tl.floor((u2 - near2 + 0.5) * 16).to(tl.int32), 0), 15
    )
    phase = (phase0 * 16 + phase1) * 16 + phase2
    eligible = (
        (precision >= 1.0 / (3.05 * 3.05))
        & (tl.abs(c0) <= 16384)
        & (tl.abs(c1) <= 16384)
        & (tl.abs(c2) <= 16384)
    )
    # Exact quarter-lattice nodes eliminate position-rounding variation. Only
    # nearest-cell subtraction rounds; at |c|,|O|<=16384 its physical error is
    # <=.001953125, covered by the independently enumerated .004 pad.
    eligible = eligible & ATLAS_GEOMETRY
    eligible = (
        eligible
        & (tl.abs(c0 - (O0 + near0 * S0)) <= 0.5 * S0 + 0.004)
        & (tl.abs(c1 - (O1 + near1 * S1)) <= 0.5 * S1 + 0.004)
        & (tl.abs(c2 - (O2 + near2 * S2)) <= 0.5 * S2 + 0.004)
    )
    packed = False
    if eligible:
        count = tl.load(AtlasCounts + phase)
        if count <= 512:
            packed = True
            lane = tl.arange(0, 512)
            code = tl.load(Atlas + phase * 512 + lane)
            r0 = near0 + (code & 15) - 4
            r1 = near1 + ((code >> 4) & 31) - 8
            r2 = near2 + ((code >> 9) & 31) - 8
            valid = (
                (lane < count)
                & (r0 >= 0)
                & (r0 < N)
                & (r1 >= 0)
                & (r1 < H)
                & (r2 >= 0)
                & (r2 < J)
            )
            _point_math(
                P,
                Norm,
                SupportFlags,
                W,
                DP,
                original_a,
                amp,
                precision,
                c0,
                c1,
                c2,
                r0,
                r1,
                r2,
                valid,
                H,
                J,
                O0,
                O1,
                O2,
                S0,
                S1,
                S2,
                LO,
                HI,
                BACK,
                SAVED_FLAGS,
                TUPLE_GRADS,
            )
    if not packed:
        _wide_math(
            P,
            Norm,
            SupportFlags,
            W,
            DP,
            Offsets,
            original_a,
            amp,
            precision,
            c0,
            c1,
            c2,
            N,
            H,
            J,
            O0,
            O1,
            O2,
            S0,
            S1,
            S2,
            LO,
            HI,
            COUNT,
            BACK,
            SAVED_FLAGS,
        )
