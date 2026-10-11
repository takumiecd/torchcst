"""Initialize an arena scratch region without mutating its saved view metadata."""

import triton as tr
import triton.language as tl


@tr.jit
def clear(Physical, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Physical + index, 0.0, index < N)
