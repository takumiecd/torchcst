"""Materialize a logical row window of mapped CST weights."""

import triton
import triton.language as tl

from prototypes.block_strip_kernels import _weight_columns_factored, _weight_lanes


@triton.jit
def materialize_logical(
    P,
    Circle,
    Section,
    Offsets,
    W,
    N: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    T: tl.constexpr,
    CG: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    PROFILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BA: tl.constexpr,
    FACTORED: tl.constexpr = False,
    ROW_GROUP_START=0,
    LOCAL_W: tl.constexpr = False,
):
    tile = tl.program_id(0)
    row_group = tile // triton.cdiv(S, BN) + ROW_GROUP_START
    local = (tile % triton.cdiv(S, BN)) * BN
    column_group = tl.program_id(1)
    start = tl.program_id(2) * BK
    station = row_group * CG + column_group
    if FACTORED:
        tl.static_assert(D == 4)
        tl.static_assert(BA == 1)
        w = _weight_columns_factored(
            P,
            Circle,
            Section,
            Offsets,
            station,
            station * S + local,
            start,
            G * S,
            T,
            G,
            S,
            PROFILE,
            BN,
            BK,
        )
    else:
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
    n = row_group * S + local + tl.arange(0, BN)
    k = column_group * T + start + tl.arange(0, BK)
    w_row = n - ROW_GROUP_START * S if LOCAL_W else n
    tl.store(
        W + w_row[:, None] * K + k[None, :],
        w,
        (n[:, None] < N)
        & (local + tl.arange(0, BN)[:, None] < S)
        & (k[None, :] < K)
        & (start + tl.arange(0, BK)[None, :] < T),
    )
