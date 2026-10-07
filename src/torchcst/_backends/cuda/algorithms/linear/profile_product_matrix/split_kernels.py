"""FP32 IEEE partial contractions and deterministic FP32 reduction."""

import triton as tr
import triton.language as tl


@tr.jit
def ieee_split_matmul(
    L,
    R,
    Partial,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    blocks: tl.constexpr = tr.cdiv(tr.cdiv(K, BK), SPLITS)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(blocks):
        kk = (split * blocks + block) * BK + k
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
        Partial + split * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@tr.jit
def reduce_partials(
    Partial, Out, COUNT: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(Partial + split * COUNT + i, i < COUNT, 0)
    tl.store(Out + i, acc, i < COUNT)
