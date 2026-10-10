"""Forward-only spatial routing; one CTA owns each batch/output tile.

Sort coarse centre bins, not per-site support IDs. Rebuild from the prepared
forward snapshot every call. Routing tensors are temporary and never retained
for backward, which still uses the original atom-owned contractions.
"""

import torch


def prepare_routing(packed, sizes, recipe):
    """Build the same coarse candidate index for fused and reusable-H routes."""
    import triton as tr

    from . import output_kernels as kernels

    _, _, no, _, lo, _, oo = sizes
    a = packed.shape[1]
    bins = tr.cdiv(no, recipe.output_tile)
    keys = torch.empty(a, device=packed.device, dtype=torch.int32)
    distances = torch.empty(tr.cdiv(a, 256), device=packed.device, dtype=torch.int32)
    kernels.routing_keys[(tr.cdiv(a, 256),)](
        packed,
        keys,
        distances,
        a,
        no,
        lo,
        oo,
        recipe.output_tile,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    sorted_keys, order = torch.sort(keys)
    boundaries = torch.searchsorted(
        sorted_keys, torch.arange(bins + 1, device=packed.device, dtype=torch.int32)
    )
    max_distance = distances.amax()
    return order, boundaries, max_distance


def h_capacity(recipe):
    from .recipe import ParallelReusedHRecipe, PreparedReusedHRecipe

    return (
        recipe.h_batch
        if type(recipe) in (ParallelReusedHRecipe, PreparedReusedHRecipe)
        else recipe.batch_tile
    )


def allocate_h(x, atom_count, recipe):
    from .recipe import ParallelReusedHRecipe, PreparedReusedHRecipe

    if type(recipe) in (ParallelReusedHRecipe, PreparedReusedHRecipe):
        # Tile-major: each atom × batch_tile slab matches the H8 baseline.
        return x.new_empty(
            (recipe.h_batch // recipe.batch_tile, atom_count, recipe.batch_tile)
        )
    return x.new_empty((atom_count, recipe.batch_tile))


def active_h_tiles(h, batch, batch_start, batch_tile):
    capacity = h.shape[0] if h.ndim == 3 else 1
    return min(capacity, (batch - batch_start + batch_tile - 1) // batch_tile)


def produce_h_chunk(x, packed, sizes, recipe, routing, h, batch_start):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, _, li, _, oi, _ = sizes
    a = packed.shape[1]
    kernels.produce_h[
        (
            tr.cdiv(a, recipe.atom_group),
            active_h_tiles(h, b, batch_start, recipe.batch_tile),
        )
    ](
        x,
        packed,
        routing[0],
        h,
        a,
        b,
        ni,
        li,
        oi,
        *x.stride(),
        batch_start,
        recipe.batch_tile,
        recipe.patch_sites,
        recipe.atom_group,
        num_warps=4,
        enable_fp_fusion=False,
    )


def prepare_output_fields(packed, routing):
    """Only the three output fields survive across H chunks, never backward."""
    import triton as tr

    from . import output_kernels as kernels

    a = packed.shape[1]
    hot = packed.new_empty((3, a))
    unsafe = torch.zeros((), device=packed.device, dtype=torch.int32)
    kernels.prepare_output_fields[(tr.cdiv(a, 256),)](
        packed, routing[0], hot, unsafe, a, 256, num_warps=4, enable_fp_fusion=False
    )
    return hot, unsafe


def aggregate_chunk(
    x, packed, sizes, recipe, routing, y, *, h=None, batch_start=0, hot=None
):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    tiles = (
        tr.cdiv(b, recipe.batch_tile)
        if h is None
        else active_h_tiles(h, b, batch_start, recipe.batch_tile)
    )
    args = (
        x,
        packed,
        *routing,
        y,
        a,
        b,
        ni,
        no,
        li,
        lo,
        oi,
        oo,
        *x.stride(),
        recipe.batch_tile,
        recipe.patch_sites,
        recipe.atom_group,
        recipe.output_tile,
    )
    options = {"H": h, "BSTART": batch_start, "num_warps": 4, "enable_fp_fusion": False}
    grid = (tr.cdiv(no, recipe.output_tile), tiles)
    if hot is None:
        kernels.output_owned[grid](*args, CACHED=h is not None, **options)
    else:
        if h is None:
            raise ValueError("prepared output ownership requires a cached H chunk")
        for fallback in (False, True):
            kernels.prepared_output_owned[grid](
                *args, Hot=hot[0], Unsafe=hot[1], FALLBACK=fallback, **options
            )


def forward_output_owned(x, packed, sizes, recipe):
    from .recipe import ParallelReusedHRecipe, PreparedReusedHRecipe, ReusedHRecipe

    b, _, no, *_ = sizes
    a = packed.shape[1]
    if not a:
        return x.new_zeros((b, no))
    y = x.new_empty((b, no))
    routing = prepare_routing(packed, sizes, recipe)
    hot = (
        prepare_output_fields(packed, routing)
        if type(recipe) is PreparedReusedHRecipe
        else None
    )
    if type(recipe) in (ReusedHRecipe, ParallelReusedHRecipe, PreparedReusedHRecipe):
        # One allocation and sequential same-stream reuse across all chunks.
        # Scratch is not saved for backward and does not scale with full batch.
        h = allocate_h(x, a, recipe)
        for batch_start in range(0, b, h_capacity(recipe)):
            produce_h_chunk(x, packed, sizes, recipe, routing, h, batch_start)
            aggregate_chunk(
                x,
                packed,
                sizes,
                recipe,
                routing,
                y,
                h=h,
                batch_start=batch_start,
                hot=hot,
            )
    else:
        aggregate_chunk(x, packed, sizes, recipe, routing, y)
    return y
