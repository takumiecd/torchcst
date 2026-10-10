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


def produce_h_chunk(x, packed, sizes, recipe, routing, h, batch_start):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, _, li, _, oi, _ = sizes
    a = packed.shape[1]
    kernels.produce_h[(tr.cdiv(a, recipe.atom_group),)](
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


def aggregate_chunk(x, packed, sizes, recipe, routing, y, *, h=None, batch_start=0):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    tiles = tr.cdiv(b, recipe.batch_tile) if h is None else 1
    kernels.output_owned[(tr.cdiv(no, recipe.output_tile), tiles)](
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
        H=h,
        BSTART=batch_start,
        CACHED=h is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )


def forward_output_owned(x, packed, sizes, recipe):
    from .recipe import ReusedHRecipe

    b, _, no, *_ = sizes
    a = packed.shape[1]
    if not a:
        return x.new_zeros((b, no))
    y = x.new_empty((b, no))
    routing = prepare_routing(packed, sizes, recipe)
    if type(recipe) is ReusedHRecipe:
        # One allocation and sequential same-stream reuse across all chunks.
        # Scratch is not saved for backward and does not scale with full batch.
        h = x.new_empty((a, recipe.batch_tile))
        for batch_start in range(0, b, recipe.batch_tile):
            produce_h_chunk(x, packed, sizes, recipe, routing, h, batch_start)
            aggregate_chunk(
                x, packed, sizes, recipe, routing, y, h=h, batch_start=batch_start
            )
    else:
        aggregate_chunk(x, packed, sizes, recipe, routing, y)
    return y
