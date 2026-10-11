"""Small routing metadata; no packed/source transpose or full-size gather."""

import triton as tr
import triton.language as tl


@tr.jit
def source_input_keys(
    Source,
    Keys,
    A: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    O: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = tl.load(Source + 4 * a + 3, a < A, O)
    phase = c - O - L * tl.floor(tl.div_rn(c - O, L))
    site = tl.minimum(tl.maximum(tl.floor(tl.div_rn(phase, L / N)), 0), N - 1)
    tl.store(Keys + a, site.to(tl.int32), a < A)


@tr.jit
def input_distances(P, Keys, Distances, A: tl.constexpr, BLOCK: tl.constexpr):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    site = tl.load(Keys + a, valid, 0)
    low = tl.load(P + 9 * A + a, valid, 0).to(tl.int32)
    high = tl.load(P + 10 * A + a, valid, 0).to(tl.int32)
    distance = tl.maximum(tl.abs(low - site), tl.abs(high - 1 - site)) + 2
    distance = tl.where(valid & (high > low), distance, 0)
    tl.store(Distances + tl.program_id(0), tl.max(distance, 0))
