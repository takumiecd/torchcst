"""Stage diagnostics for the fused block kernel. Host launch stays outside."""

import triton as tr
import triton.language as tl

from prototypes.block_strip_kernels import _weight_lanes


@tr.jit
def fused_stage(
    X,
    P,
    Circle,
    Section,
    Offsets,
    Y,
    Sink,
    W,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    STAGE: tl.constexpr,
):
    """Same grid as block_fused. STAGE 0 matches LATE_REDUCE; 1 eval; 2 dot."""
    tile = tl.program_id(1)
    r = tile // tr.cdiv(S, BN)
    local = (tile % tr.cdiv(S, BN)) * BN
    n = r * S + local + tl.arange(0, BN)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    if STAGE == 0:
        acc = tl.full((BM, BN), 0, tl.float32)
        for c in range(CG):
            station = r * CG + c
            for start in range(0, T, BK):
                k = start + tl.arange(0, BK)
                x = tl.load(
                    X + m[:, None] * K + (c * T + k)[None, :],
                    (m[:, None] < M) & (k[None, :] < T) & (c * T + k[None, :] < K),
                    0.0,
                )
                w = _weight_lanes(
                    P,
                    Circle,
                    Section,
                    Offsets,
                    station,
                    station * S + local,
                    start,
                    G * S,
                    T,
                    D,
                    G,
                    S,
                    PROFILE,
                    BN,
                    BK,
                    BA,
                )
                acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
        tl.store(
            Y + m[:, None] * N + n[None, :],
            acc,
            (m[:, None] < M)
            & (n[None, :] < N)
            & (local + tl.arange(0, BN)[None, :] < S),
        )
    elif STAGE == 1:
        total = 0.0
        for c in range(CG):
            station = r * CG + c
            for start in range(0, T, BK):
                w = _weight_lanes(
                    P,
                    Circle,
                    Section,
                    Offsets,
                    station,
                    station * S + local,
                    start,
                    G * S,
                    T,
                    D,
                    G,
                    S,
                    PROFILE,
                    BN,
                    BK,
                    BA,
                )
                total += tl.sum(w)
        tl.store(Sink + tl.program_id(0) * tl.num_programs(1) + tile, total)
    elif STAGE == 2:
        frag = tl.arange(0, 16)
        w = tl.load(W + frag[:, None] * 16 + frag[None, :])
        acc = tl.full((BM, BN), 0, tl.float32)
        for c in range(CG):
            for start in range(0, T, BK):
                k = start + tl.arange(0, BK)
                x = tl.load(
                    X + m[:, None] * K + (c * T + k)[None, :],
                    (m[:, None] < M) & (k[None, :] < T) & (c * T + k[None, :] < K),
                    0.0,
                )
                acc = tl.dot(x, tl.trans(w), acc, input_precision="ieee")
        tl.store(
            Y + m[:, None] * N + n[None, :],
            acc,
            (m[:, None] < M)
            & (n[None, :] < N)
            & (local + tl.arange(0, BN)[None, :] < S),
        )
