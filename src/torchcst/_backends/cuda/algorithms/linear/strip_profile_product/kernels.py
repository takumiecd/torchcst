"""Restrict globally normalized metadata to a physical input tile."""

import triton as tr
import triton.language as tl


@tr.jit
def restrict_tile(
    Global,
    Local,
    Pitch,
    A: tl.constexpr,
    TILE: tl.constexpr,
    TILE_ID: tl.constexpr,
    COUNT: tl.constexpr,
    NO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = a < A
    lo = tl.minimum(
        tl.maximum(tl.load(Global + 9 * A + a, valid, 0) - TILE_ID * TILE, 0), COUNT
    )
    hi = tl.minimum(
        tl.maximum(tl.load(Global + 10 * A + a, valid, 0) - TILE_ID * TILE, 0), COUNT
    )
    out_lo = tl.load(Global + 11 * A + a, valid, NO)
    out_hi = tl.load(Global + 12 * A + a, valid, 0)
    live = (hi > lo) & (out_hi > out_lo)
    for f in tl.static_range(9):
        v = tl.load(Global + f * A + a, valid, 0)
        if f == 2:
            v -= TILE_ID * tl.load(Pitch)
        if f == 8:
            v = tl.where(live, v, 0)
        tl.store(Local + f * A + a, v, valid)
    tl.store(Local + 9 * A + a, tl.where(live, lo, COUNT), valid)
    tl.store(Local + 10 * A + a, tl.where(live, hi, 0), valid)
    tl.store(Local + 11 * A + a, tl.where(live, out_lo, NO), valid)
    tl.store(Local + 12 * A + a, tl.where(live, out_hi, 0), valid)
