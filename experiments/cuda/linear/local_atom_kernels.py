"""Direct atom/site contractions. No W/dW tile and no global atomics."""

import triton as tr
import triton.language as tl

from torchcst.nn._backends._triton_kernels import _profile


@tr.jit
def _bucket(station, slot: tl.constexpr, G: tl.constexpr):
    if G == 1:
        return 0
    return tl.where(slot == 0, 2 * ((station + G - 1) % G) + 1, 2 * station + slot - 1)


@tr.jit
def _row_possible(P, Bounds, a, cosine, sine, D: tl.constexpr):
    """Distance to a box enclosing this row's sites; conservative row cull."""
    low_r, high_r = tl.load(Bounds), tl.load(Bounds + D - 1)
    lower = 0.0
    for dim in tl.static_range(D):
        if dim < 2:
            direction = cosine if dim == 0 else sine
            low = tl.minimum(low_r * direction, high_r * direction)
            high = tl.maximum(low_r * direction, high_r * direction)
        else:
            low = tl.load(Bounds + dim - 1)
            high = tl.load(Bounds + D - 1 + dim - 1)
        center = tl.load(P + a * (D + 2) + 2 + dim)
        gap = tl.maximum(tl.maximum(low - center, center - high), 0.0)
        lower += gap * gap
    precision = tl.load(P + a * (D + 2) + 1)
    # Slack only admits extra work; it does not change the evaluated profile.
    return lower * precision <= 1.00001


@tr.jit
def _row_values(
    P,
    Section,
    a,
    cosine,
    sine,
    k,
    K: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
):
    rho = tl.load(Section + k * (D - 1), k < K, 0.0)
    squared = tl.full(k.shape, 0, tl.float32)
    for dim in tl.static_range(D):
        if dim == 0:
            site = rho * cosine
        elif dim == 1:
            site = rho * sine
        else:
            site = tl.load(Section + k * (D - 1) + dim - 1, k < K, 0.0)
        center = tl.load(P + a * (D + 2) + 2 + dim)
        diff = site - center
        squared += diff * diff
    precision = tl.load(P + a * (D + 2) + 1)
    value, slope = _profile(squared, precision, PROFILE)
    return tl.where(k < K, value, 0.0), tl.where(k < K, slope, 0.0)


@tr.jit
def direct_forward(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    station = n // S
    cosine, sine = tl.load(Circle + 2 * n), tl.load(Circle + 2 * n + 1)
    acc = tl.full((BM,), 0, tl.float32)
    for slot in tl.static_range(1 if G == 1 else 3):
        bucket = _bucket(station, slot, G)
        begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
        for a in range(begin, end):
            amplitude = tl.load(P + a * (D + 2))
            if (amplitude != 0) & _row_possible(P, Bounds, a, cosine, sine, D):
                for start in range(tr.cdiv(K, BK)):
                    k = start * BK + tl.arange(0, BK)
                    value, _ = _row_values(
                        P, Section, a, cosine, sine, k, K, D, PROFILE
                    )
                    active = (k < K) & (value != 0)
                    if tl.sum(active.to(tl.int32), 0) > 0:
                        x = tl.load(
                            X + m[:, None] * K + k[None, :],
                            (m[:, None] < M) & active[None, :],
                            0.0,
                        )
                        acc += tl.sum(x * value[None, :], 1) * amplitude
    tl.store(Y + m * N + n, acc, m < M)


@tr.jit
def direct_dx(
    DY,
    P,
    Circle,
    Section,
    Offsets,
    DX,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    RN: tl.constexpr,
):
    k = tl.program_id(1)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    rho = tl.load(Section + k * (D - 1))
    acc = tl.full((BM,), 0, tl.float32)
    for station in range(G):
        n = station * S + tl.arange(0, RN)
        valid = (n < N) & (tl.arange(0, RN) < S)
        sx = tl.load(Circle + 2 * n, valid, 0.0) * rho
        sy = tl.load(Circle + 2 * n + 1, valid, 0.0) * rho
        for slot in tl.static_range(1 if G == 1 else 3):
            bucket = _bucket(station, slot, G)
            begin, end = tl.load(Offsets + bucket), tl.load(Offsets + bucket + 1)
            for a in range(begin, end):
                squared = tl.full((RN,), 0, tl.float32)
                for dim in tl.static_range(D):
                    if dim == 0:
                        site = sx
                    elif dim == 1:
                        site = sy
                    else:
                        site = tl.load(Section + k * (D - 1) + dim - 1)
                    center = tl.load(P + a * (D + 2) + 2 + dim)
                    diff = site - center
                    squared += diff * diff
                value, _ = _profile(squared, tl.load(P + a * (D + 2) + 1), PROFILE)
                amplitude = tl.load(P + a * (D + 2))
                active = valid & (value != 0) & (amplitude != 0)
                if tl.sum(active.to(tl.int32), 0) > 0:
                    dy = tl.load(
                        DY + m[:, None] * N + n[None, :],
                        (m[:, None] < M) & active[None, :],
                        0.0,
                    )
                    acc += (
                        tl.sum(dy * tl.where(active, value, 0.0)[None, :], 1)
                        * amplitude
                    )
    tl.store(DX + m * K + k, acc, m < M)


@tr.jit
def direct_dp(
    X,
    DY,
    P,
    Circle,
    Section,
    Offsets,
    Bounds,
    DP,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    S: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BG: tl.constexpr,
    DD: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.arange(0, BG)
    end = tl.load(Offsets + b + 1, b < 2 * G + 1, 2147483647)
    bucket = tl.sum(((b < 2 * G + 1) & (a >= end)).to(tl.int32), 0)
    amplitude = tl.load(P + a * (D + 2))
    da = 0.0
    dims = tl.arange(0, DD)
    dc = tl.full((DD,), 0, tl.float32)
    if bucket < 2 * G:
        for target in range(1 + bucket % 2):
            station = (bucket // 2 + target) % G
            for local in range(S):
                n = station * S + local
                if n < N:
                    cosine, sine = tl.load(Circle + 2 * n), tl.load(Circle + 2 * n + 1)
                    if _row_possible(P, Bounds, a, cosine, sine, D):
                        for start in range(tr.cdiv(K, BK)):
                            k = start * BK + tl.arange(0, BK)
                            value, slope = _row_values(
                                P, Section, a, cosine, sine, k, K, D, PROFILE
                            )
                            active = (k < K) & ((value != 0) | (slope != 0))
                            if tl.sum(active.to(tl.int32), 0) > 0:
                                dot = tl.full((BK,), 0, tl.float32)
                                for mb in range(tr.cdiv(M, BM)):
                                    m = mb * BM + tl.arange(0, BM)
                                    x = tl.load(
                                        X + m[:, None] * K + k[None, :],
                                        (m[:, None] < M) & active[None, :],
                                        0.0,
                                    )
                                    dy = tl.load(DY + m * N + n, m < M, 0.0)
                                    dot += tl.sum(x * dy[:, None], 0)
                                da += tl.sum(dot * value, 0)
                                scale = -2 * amplitude * dot * slope
                                rho = tl.load(Section + k * (D - 1), k < K, 0.0)
                                extra = tl.load(
                                    Section + k[:, None] * (D - 1) + dims[None, :] - 1,
                                    (k[:, None] < K)
                                    & (dims[None, :] >= 2)
                                    & (dims[None, :] < D),
                                    0.0,
                                )
                                sites = tl.where(
                                    dims[None, :] == 0,
                                    rho[:, None] * cosine,
                                    tl.where(
                                        dims[None, :] == 1, rho[:, None] * sine, extra
                                    ),
                                )
                                center = tl.load(
                                    P + a * (D + 2) + 2 + dims, dims < D, 0.0
                                )
                                dc += tl.sum(
                                    scale[:, None] * (sites - center[None, :]), 0
                                )
    tl.store(DP + a * (D + 2), da)
    tl.store(DP + a * (D + 2) + 1, 0.0)
    tl.store(DP + a * (D + 2) + 2 + dims, dc, dims < D)
