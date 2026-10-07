"""Lossless support-tile keys and packed-view permutation, rebuilt each forward."""

import triton as tr
import triton.language as tl


@tr.jit
def support_keys(
    P,
    Keys,
    A: tl.constexpr,
    NI: tl.constexpr,
    NO: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = a < A
    ilo = tl.load(P + 9 * A + a, live, 0).to(tl.int32)
    olo = tl.load(P + 11 * A + a, live, 0).to(tl.int32)
    ti = tl.minimum(tl.maximum(ilo, 0), NI - 1) // TILE
    to = tl.minimum(tl.maximum(olo, 0), NO - 1) // TILE
    owner = to.to(tl.int64) * tr.cdiv(NI, TILE) + ti.to(tl.int64)
    # Canonical IDs occupy the low32 bits. Cast before multiplication so that
    # large charts never overflow an intermediate int32 key.
    key = owner * 4294967296 + a.to(tl.int64)
    tl.store(Keys + a, key, live)


@tr.jit
def copy_ordered(
    P,
    Keys,
    Ordered,
    Order,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = a < A
    key = tl.load(Keys + a, live, 0)
    original = (key & 4294967295).to(tl.int32)
    tl.store(Order + a, original, live)
    for field in tl.static_range(13):
        value = tl.load(P + field * A + original, live, 0)
        tl.store(Ordered + field * A + a, value, live)


def order_support(packed, ni, no, tile):
    """Return a lossless packed view and a saved physical-to-canonical map."""
    import torch

    a = packed.shape[1]
    order = torch.empty(a, dtype=torch.int32, device=packed.device)
    if not a:
        return packed, order
    keys = torch.empty(a, dtype=torch.int64, device=packed.device)
    support_keys[(tr.cdiv(a, 256),)](packed, keys, a, ni, no, tile, 256, num_warps=4)
    keys = keys.sort().values
    ordered = torch.empty_like(packed)
    copy_ordered[(tr.cdiv(a, 256),)](
        packed,
        keys,
        ordered,
        order,
        a,
        256,
        num_warps=4,
    )
    return ordered, order
