"""Reduce bounded native hot partial sums and reset for the next batch chunk."""

import triton as tr
import triton.language as tl


@tr.jit
def finish(
    Y,
    Cold,
    Hot,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    SHARDS: tl.constexpr,
    BLOCK: tl.constexpr,
    POINT_MAJOR: tl.constexpr = False,
    ROWS: tl.constexpr = 0,
    SITES: tl.constexpr = 0,
):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, SHARDS)
    index = (j % SITES) * ROWS + j // SITES if POINT_MAJOR else j
    addr = Hot + s[:, None] * STRIDE + index[None, :]
    partial = tl.load(addr, j[None, :] < N, 0).to(tl.float64)
    value = tl.load(Cold + j, j < N, 0) + tl.sum(partial, 0)
    previous = tl.load(Y + j, j < N, 0).to(tl.float64)
    tl.store(Y + j, previous + value, j < N)
    tl.store(addr, 0.0, j[None, :] < N)
