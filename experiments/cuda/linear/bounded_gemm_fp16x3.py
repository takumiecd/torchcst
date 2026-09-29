"""Experimental local-window FP32 GEMM via three FP16 Tensor Core products."""

import triton as tr
import triton.language as tl


@tr.jit
def bounded_gemm_fp16x3_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSK: tl.constexpr,
    BSN: tl.constexpr,
    CSM: tl.constexpr,
    CSN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ADD: tl.constexpr,
    RESIDUAL_PRODUCT: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    correction = tl.full((BM, BN), 0, tl.float32)
    if RESIDUAL_PRODUCT:
        low_product = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tr.cdiv(K, BK)):
        ks = offset * BK + inner
        a = tl.load(
            A + rows[:, None] * ASM + ks[None, :] * ASK,
            (rows[:, None] < M) & (ks[None, :] < K),
            0,
        )
        b = tl.load(
            B + ks[:, None] * BSK + cols[None, :] * BSN,
            (ks[:, None] < K) & (cols[None, :] < N),
            0,
        )
        ah = a.to(tl.float16)
        bh = b.to(tl.float16)
        al = ((a - ah.to(tl.float32)) * 4096.0).to(tl.float16)
        bl = ((b - bh.to(tl.float32)) * 4096.0).to(tl.float16)
        acc = tl.dot(ah, bh, acc)
        correction = tl.dot(ah, bl, correction)
        correction = tl.dot(al, bh, correction)
        if RESIDUAL_PRODUCT:
            low_product = tl.dot(al, bl, low_product)
    acc += correction * (1.0 / 4096.0)
    if RESIDUAL_PRODUCT:
        acc += low_product * (1.0 / 16777216.0)
    ptr = C + rows[:, None] * CSM + cols[None, :] * CSN
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    if ADD:
        acc += tl.load(ptr, mask, 0)
    tl.store(ptr, acc, mask)


def bounded_gemm_fp16x3(a, b, out, *, add=False, residual_product=False):
    """Multiply two FP32 tensors using only register-local FP16 splits."""
    if a.ndim != 2 or b.ndim != 2 or out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("invalid bounded GEMM shapes")
    if a.shape[1] != b.shape[0] or out.stride(1) != 1:
        raise ValueError("invalid bounded GEMM strides")
    bm, bn, bk = 64, 128, 32
    bounded_gemm_fp16x3_kernel[(tr.cdiv(a.shape[0], bm), tr.cdiv(b.shape[1], bn))](
        a,
        b,
        out,
        a.shape[0],
        b.shape[1],
        a.shape[1],
        *a.stride(),
        *b.stride(),
        *out.stride(),
        bm,
        bn,
        bk,
        add,
        residual_product,
        num_warps=4,
    )
