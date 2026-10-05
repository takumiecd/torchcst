"""Validated rolling-window CUDA kernels; whole-support normalization/VJP."""

import triton as tr
import triton.language as tl

COMPILED_KERNELS = {}
_ATLAS_CACHE = {}


def atlas_tables(device):
    key = str(device)
    if key not in _ATLAS_CACHE:
        from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip._shared.atlas_table import (
            build_atlas,
        )

        (codes, counts) = build_atlas()
        _ATLAS_CACHE[key] = (
            codes.to(device),
            counts.to(device),
        )
    return _ATLAS_CACHE[key]


@tr.jit
def _read_dw(
    W, Halo, Start, r0, r1, r2, H: tl.constexpr, J: tl.constexpr, valid, R: tl.constexpr
):
    current = tl.load(
        W + (r0 - Start) * (H * J) + r1 * J + r2,
        valid & (r0 >= Start) & (r0 < Start + R),
        0.0,
    )
    previous = tl.load(
        Halo + (r0 - (Start - 8)) * (H * J) + r1 * J + r2,
        valid & (r0 < Start) & (r0 >= Start - 8),
        0.0,
    )
    return tl.where(r0 < Start, previous, current)


@tr.jit
def _sum_norm_count(n0, c0, n1, c1):
    return (n0 + n1, c0 + c1)


@tr.jit
def _sum_four(a0, b0, c0, d0, a1, b1, c1, d1):
    return (a0 + a1, b0 + b1, c0 + c1, d0 + d1)


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
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
):
    d0 = O0 + r0 * S0 - c0
    d1 = O1 + r1 * S1 - c1
    d2 = O2 + r2 * S2 - c2
    q = (d0 * d0 + d1 * d1 + d2 * d2) * precision
    t = tl.where(valid, tl.maximum(1.0 - q, 0.0), 0.0)
    k = t * t * t
    if not BACK:
        if SAVED_FLAGS:
            (norm2, positive) = tl.reduce(
                (k * k, (k > 0).to(tl.int32)), 0, _sum_norm_count
            )
        else:
            norm2 = tl.sum(k * k, 0)
        raw_norm = tl.sqrt_rn(norm2)
        norm = tl.maximum(raw_norm, 1e-06)
        tl.store(Norm + original_a, norm, True & (MODE == 0))
        if SAVED_FLAGS:
            live = raw_norm >= 1e-06
            singleton = (positive == 1) & live
            flags = live.to(tl.int32) | singleton.to(tl.int32) << 1
            tl.store(SupportFlags + original_a, flags, True & (MODE == 0))
        if MODE == 1:
            norm = tl.load(Norm + original_a)
        effective_amp = amp / norm
        tl.atomic_add(
            W + (r0 - RowStart) * (H * J) + r1 * J + r2,
            effective_amp * k,
            valid & (k > 0) & (MODE == 1) & (r0 >= RowStart) & (r0 < RowStart + R),
            sem="relaxed",
        )
    else:
        dw = _read_dw(W, Halo, RowStart, r0, r1, r2, H, J, valid & (k > 0), R)
        norm = tl.load(Norm + original_a)
        if MODE == 1:
            norm = tl.load(Norm + original_a)
        effective_amp = amp / norm
        g0 = tl.sum(dw * k, 0)
        if SAVED_FLAGS:
            flags = tl.load(SupportFlags + original_a).to(tl.int32)
            live_norm = flags & 1 != 0
            singleton = flags & 2 != 0
        else:
            norm2 = tl.sum(k * k, 0)
            live_norm = tl.sqrt_rn(norm2) >= 1e-06
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
            (gl_sum, gc0, gc1, gc2) = tl.reduce(
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
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
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
        r1 = near1 + (code >> 4 & 31) - 8
        r2 = near2 + (code >> 9 & 31) - 8
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
            dw = _read_dw(W, Halo, RowStart, r0, r1, r2, H, J, valid & (k > 0), R)
            g0 += tl.sum(dw * k, 0)
    if BACK:
        norm = tl.load(Norm + original_a)
        if SAVED_FLAGS:
            flags = tl.load(SupportFlags + original_a).to(tl.int32)
            live = flags & 1 != 0
            singleton = flags & 2 != 0
        else:
            live = tl.sqrt_rn(norm2) >= 1e-06
            singleton = (positive == 1) & live
    else:
        raw_norm = tl.sqrt_rn(norm2)
        norm = tl.maximum(raw_norm, 1e-06)
        live = raw_norm >= 1e-06
        singleton = (positive == 1) & live
        tl.store(Norm + original_a, norm, True & (MODE == 0))
        if SAVED_FLAGS:
            tl.store(
                SupportFlags + original_a,
                live.to(tl.int32) | singleton.to(tl.int32) << 1,
                True & (MODE == 0),
            )
    if MODE == 1:
        norm = tl.load(Norm + original_a)
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
        r1 = near1 + (code >> 4 & 31) - 8
        r2 = near2 + (code >> 9 & 31) - 8
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
        if not BACK:
            tl.atomic_add(
                W + (r0 - RowStart) * (H * J) + r1 * J + r2,
                effective_amp * k,
                valid & (k > 0) & (MODE == 1) & (r0 >= RowStart) & (r0 < RowStart + R),
                sem="relaxed",
            )
        else:
            dw = _read_dw(W, Halo, RowStart, r0, r1, r2, H, J, valid & (k > 0), R)
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


_NARROW_CACHE = {}


def narrow_tables(device):
    key = str(device)
    if key not in _NARROW_CACHE:
        from torchcst._backends.cuda.algorithms.linear.normalized_euclidean_strip.window.narrow_table import (
            build_narrow_tables,
        )

        (codes, counts) = build_narrow_tables()
        _NARROW_CACHE[key] = (
            codes.to(device),
            counts.to(device),
        )
    return _NARROW_CACHE[key]


@tr.jit
def _narrow_math(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Narrow,
    NarrowCounts,
    original_a,
    amp,
    precision,
    c0,
    c1,
    c2,
    near0,
    near1,
    near2,
    phase,
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
    BACK: tl.constexpr,
    SAVED_FLAGS: tl.constexpr,
    TUPLE_GRADS: tl.constexpr,
    CAP: tl.constexpr,
    BASE: tl.constexpr,
    BAND: tl.constexpr,
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
):
    lane = tl.arange(0, CAP)
    count = tl.load(NarrowCounts + BAND * 4096 + phase)
    code = tl.load(Narrow + BASE + phase * CAP + lane, lane < count, 0)
    r0 = near0 + (code & 15) - 4
    r1 = near1 + (code >> 4 & 31) - 8
    r2 = near2 + (code >> 9 & 31) - 8
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
        RowStart=RowStart,
        Halo=Halo,
        MODE=MODE,
        R=R,
    )


@tr.jit
def _atom_body(
    Program,
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Order,
    Offsets,
    Atlas,
    AtlasCounts,
    Narrow,
    NarrowCounts,
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
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
):
    a = Program
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
    eligible = eligible & ATLAS_GEOMETRY
    eligible = (
        eligible
        & (tl.abs(c0 - (O0 + near0 * S0)) <= 0.5 * S0 + 0.004)
        & (tl.abs(c1 - (O1 + near1 * S1)) <= 0.5 * S1 + 0.004)
        & (tl.abs(c2 - (O2 + near2 * S2)) <= 0.5 * S2 + 0.004)
    )
    packed = False
    if eligible and precision >= 25.0:
        lane = tl.arange(0, 32)
        r0 = near0 + lane * 0
        r1 = near1 + lane * 0
        r2 = near2 + lane * 0
        valid = (
            (lane == 0)
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
            RowStart=RowStart,
            Halo=Halo,
            MODE=MODE,
            R=R,
        )
        packed = True
    elif eligible and precision >= 11.112:
        _narrow_math(
            P,
            Norm,
            SupportFlags,
            W,
            DP,
            Narrow,
            NarrowCounts,
            original_a,
            amp,
            precision,
            c0,
            c1,
            c2,
            near0,
            near1,
            near2,
            phase,
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
            BACK,
            SAVED_FLAGS,
            TUPLE_GRADS,
            CAP=2,
            BASE=0,
            BAND=0,
            RowStart=RowStart,
            Halo=Halo,
            MODE=MODE,
            R=R,
        )
        packed = True
    elif eligible and precision >= 2.367:
        _narrow_math(
            P,
            Norm,
            SupportFlags,
            W,
            DP,
            Narrow,
            NarrowCounts,
            original_a,
            amp,
            precision,
            c0,
            c1,
            c2,
            near0,
            near1,
            near2,
            phase,
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
            BACK,
            SAVED_FLAGS,
            TUPLE_GRADS,
            CAP=8,
            BASE=8192,
            BAND=1,
            RowStart=RowStart,
            Halo=Halo,
            MODE=MODE,
            R=R,
        )
        packed = True
    elif eligible and precision >= 0.827:
        _narrow_math(
            P,
            Norm,
            SupportFlags,
            W,
            DP,
            Narrow,
            NarrowCounts,
            original_a,
            amp,
            precision,
            c0,
            c1,
            c2,
            near0,
            near1,
            near2,
            phase,
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
            BACK,
            SAVED_FLAGS,
            TUPLE_GRADS,
            CAP=32,
            BASE=40960,
            BAND=2,
            RowStart=RowStart,
            Halo=Halo,
            MODE=MODE,
            R=R,
        )
        packed = True
    elif eligible:
        count = tl.load(AtlasCounts + phase)
        if count <= 512:
            packed = True
            lane = tl.arange(0, 512)
            code = tl.load(Atlas + phase * 512 + lane)
            r0 = near0 + (code & 15) - 4
            r1 = near1 + (code >> 4 & 31) - 8
            r2 = near2 + (code >> 9 & 31) - 8
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
                RowStart=RowStart,
                Halo=Halo,
                MODE=MODE,
                R=R,
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
            RowStart=RowStart,
            Halo=Halo,
            MODE=MODE,
            R=R,
        )


@tr.jit
def _persistent_atoms(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Order,
    Offsets,
    Atlas,
    AtlasCounts,
    Narrow,
    NarrowCounts,
    Total,
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
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
    Prefix,
):
    if MODE == 0:
        begin = 0
        end = tl.load(Total)
    else:
        groups = (N + 31) // 32
        low = tl.maximum(RowStart // 32 - 1, 0)
        high = tl.minimum((RowStart + R + 31) // 32 + 1, groups)
        begin = tl.load(Prefix + low)
        end = tl.load(Prefix + high)
        if tl.load(Total) == 0:
            end = begin
    pos = begin + tl.program_id(0)
    while pos < end:
        original = tl.load(Order + pos)
        c0 = tl.load(P + original * 5 + 2)
        c1 = tl.load(P + original * 5 + 3)
        c2 = tl.load(P + original * 5 + 4)
        log = tl.load(P + original * 5 + 1)
        precision = tl.exp(-2.0 * tl.minimum(tl.maximum(log, LO), HI))
        near0 = tl.floor((c0 - O0) / S0 + 0.5).to(tl.int32)
        near1 = tl.floor((c1 - O1) / S1 + 0.5).to(tl.int32)
        near2 = tl.floor((c2 - O2) / S2 + 0.5).to(tl.int32)
        singleton = (
            ATLAS_GEOMETRY
            & (precision >= 25.0)
            & (tl.abs(c0) <= 16384)
            & (tl.abs(c1) <= 16384)
            & (tl.abs(c2) <= 16384)
        )
        singleton = (
            singleton
            & (tl.abs(c0 - (O0 + near0 * S0)) <= 0.5 * S0 + 0.004)
            & (tl.abs(c1 - (O1 + near1 * S1)) <= 0.5 * S1 + 0.004)
            & (tl.abs(c2 - (O2 + near2 * S2)) <= 0.5 * S2 + 0.004)
        )
        owner = tl.minimum(tl.maximum((near0 + 4) // R, 0), (N - 1) // R)
        if MODE == 0 or ~singleton & (MODE == 1 or owner == RowStart // R):
            _atom_body(
                pos,
                P,
                Norm,
                SupportFlags,
                W,
                DP,
                Order,
                Offsets,
                Atlas,
                AtlasCounts,
                Narrow,
                NarrowCounts,
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
                B0,
                B1,
                B2,
                B,
                BACK,
                SORTED,
                BALL,
                COUNT,
                SAVED_FLAGS,
                TUPLE_GRADS,
                ATLAS_GEOMETRY,
                RowStart=RowStart,
                Halo=Halo,
                MODE=MODE,
                R=R,
            )
        pos += tl.num_programs(0)


@tr.jit
def _packed_singletons(
    P,
    Norm,
    SupportFlags,
    W,
    DP,
    Order,
    Offsets,
    Atlas,
    AtlasCounts,
    Narrow,
    NarrowCounts,
    Total,
    A: tl.constexpr,
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
    RowStart,
    Halo,
    MODE: tl.constexpr,
    R: tl.constexpr,
    Prefix,
):
    if MODE == 0:
        begin = 0
        end = A
    else:
        groups = (N + 31) // 32
        low = tl.maximum(RowStart // 32 - 1, 0)
        high = tl.minimum((RowStart + R + 31) // 32 + 1, groups)
        begin = tl.load(Prefix + low)
        end = tl.load(Prefix + high)
        if tl.load(Total) == A:
            end = begin
    pos = begin + tl.program_id(0) * 256
    while pos < end:
        position = pos + tl.arange(0, 256)
        if MODE == 0:
            lane = position
        else:
            lane = tl.load(Order + position, position < end, 0)
        mask = (position < end) & (lane < A)
        amp = tl.load(P + 5 * lane, mask, 0.0)
        log = tl.load(P + 5 * lane + 1, mask, 0.0)
        precision = tl.exp(-2.0 * tl.minimum(tl.maximum(log, LO), HI))
        c0 = tl.load(P + 5 * lane + 2, mask, 0.0)
        c1 = tl.load(P + 5 * lane + 3, mask, 0.0)
        c2 = tl.load(P + 5 * lane + 4, mask, 0.0)
        r0 = tl.floor((c0 - O0) / S0 + 0.5).to(tl.int32)
        r1 = tl.floor((c1 - O1) / S1 + 0.5).to(tl.int32)
        r2 = tl.floor((c2 - O2) / S2 + 0.5).to(tl.int32)
        eligible = (
            mask
            & ATLAS_GEOMETRY
            & (precision >= 25.0)
            & (tl.abs(c0) <= 16384)
            & (tl.abs(c1) <= 16384)
            & (tl.abs(c2) <= 16384)
        )
        eligible = (
            eligible
            & (tl.abs(c0 - (O0 + r0 * S0)) <= 0.5 * S0 + 0.004)
            & (tl.abs(c1 - (O1 + r1 * S1)) <= 0.5 * S1 + 0.004)
            & (tl.abs(c2 - (O2 + r2 * S2)) <= 0.5 * S2 + 0.004)
        )
        valid = (
            eligible
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
        if not BACK:
            raw = tl.sqrt_rn(k * k)
            norm = tl.maximum(raw, 1e-06)
            tl.store(Norm + lane, norm, eligible & (MODE == 0))
            live = raw >= 1e-06
            singleton = (k > 0) & live
            if SAVED_FLAGS:
                tl.store(
                    SupportFlags + lane,
                    live.to(tl.int32) | singleton.to(tl.int32) << 1,
                    eligible & (MODE == 0),
                )
            if MODE == 1:
                norm = tl.load(Norm + lane, eligible, 1.0)
            tl.atomic_add(
                W + (r0 - RowStart) * (H * J) + r1 * J + r2,
                amp / norm * k,
                valid & (k > 0) & (MODE == 1) & (r0 >= RowStart) & (r0 < RowStart + R),
                sem="relaxed",
            )
            fallback = mask & ~eligible
            amount = tl.sum(fallback.to(tl.int32), 0)
            if MODE == 0 and amount > 0:
                base = tl.atomic_add(Total, amount, sem="relaxed")
                position = base + tl.cumsum(fallback.to(tl.int32), 0) - 1
                tl.store(Order + position, lane, fallback)
        else:
            dw = _read_dw(
                W,
                Halo,
                RowStart,
                r0,
                r1,
                r2,
                H,
                J,
                valid
                & (k > 0)
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
                R,
            )
            norm = tl.load(Norm + lane, eligible, 1.0)
            if SAVED_FLAGS:
                flags = tl.load(SupportFlags + lane, eligible, 0).to(tl.int32)
                live = flags & 1 != 0
                singleton = flags & 2 != 0
            else:
                live = tl.sqrt_rn(k * k) >= 1e-06
                singleton = (k > 0) & live
            g0 = dw * k
            gauge = tl.where(live, g0 / (norm * norm), 0.0)
            residual = dw - gauge * k
            effective_amp = amp / norm
            dc = tl.where(
                singleton, 0.0, effective_amp * 6.0 * precision * t * t * residual
            )
            gl = tl.where(
                singleton | (k == 0), 0.0, effective_amp * 6.0 * q * t * t * residual
            )
            tl.store(
                DP + 5 * lane,
                g0 / norm,
                eligible
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
            )
            tl.store(
                DP + 5 * lane + 1,
                tl.where((log >= LO) & (log <= HI), gl, 0.0),
                eligible
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
            )
            tl.store(
                DP + 5 * lane + 2,
                dc * d0,
                eligible
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
            )
            tl.store(
                DP + 5 * lane + 3,
                dc * d1,
                eligible
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
            )
            tl.store(
                DP + 5 * lane + 4,
                dc * d2,
                eligible
                & (tl.minimum(tl.maximum(r0 // R, 0), (N - 1) // R) == RowStart // R),
            )
        pos += tl.num_programs(0) * 256


@tr.jit
def _hist(Keys, Counts, A: tl.constexpr):
    lane = tl.program_id(0) * 256 + tl.arange(0, 256)
    key = tl.load(Keys + lane, lane < A, 0)
    tl.atomic_add(Counts + key, 1, lane < A, sem="relaxed")


@tr.jit
def _scatter(Keys, Cursors, Order, A: tl.constexpr):
    lane = tl.program_id(0) * 256 + tl.arange(0, 256)
    key = tl.load(Keys + lane, lane < A, 0)
    pos = tl.atomic_add(Cursors + key, 1, lane < A, sem="relaxed")
    tl.store(Order + pos, lane, lane < A)
