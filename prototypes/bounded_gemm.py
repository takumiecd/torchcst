"""FP32-output GEMM for one bounded logical weight window."""

import triton as tr
import triton.language as tl


@tr.jit
def bounded_gemm_kernel(
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
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
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
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    ptr = C + rows[:, None] * CSM + cols[None, :] * CSN
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    if ADD:
        acc += tl.load(ptr, mask, 0)
    tl.store(ptr, acc, mask)


def bounded_gemm(a, b, out, *, add=False):
    """Multiply two 2D FP32 tensors into a row-major view."""
    if a.ndim != 2 or b.ndim != 2 or out.shape != (a.shape[0], b.shape[1]):
        raise ValueError("invalid bounded GEMM shapes")
    if a.shape[1] != b.shape[0] or out.stride(1) != 1:
        raise ValueError("invalid bounded GEMM strides")
    bm, bn, bk = 32, 128, 32
    bounded_gemm_kernel[(tr.cdiv(a.shape[0], bm), tr.cdiv(b.shape[1], bn))](
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
        num_warps=4,
    )
