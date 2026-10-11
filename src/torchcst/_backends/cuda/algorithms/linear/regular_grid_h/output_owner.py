"""Forward-only spatial routing; one CTA owns each batch/output tile.

Sort centre bins or centre sites, never per-site support IDs. Rebuild from the
prepared forward snapshot every call. Forward routing tensors are temporary
and never retained for backward.
"""

import torch


def prepare_routing(packed, sizes, recipe, *, output=True, tile=None):
    """Build ephemeral centre prefixes; backward variants choose input keys."""
    import triton as tr

    from . import output_kernels as kernels

    _, ni, no, li, lo, oi, oo = sizes
    secondary_n, secondary_l, secondary_o = no, lo, oo
    if not output:
        no, lo, oo = ni, li, oi
    bo = recipe.output_tile if tile is None else tile
    a = packed.shape[1]
    site_routing = (
        uses_site_routing(recipe) if output else uses_input_site_routing(recipe)
    )
    secondary_order = not output and uses_input_secondary_order(recipe)
    routing_options = (
        {
            "SECONDARY_OUTPUT": True,
            "SECONDARY_N": secondary_n,
            "SECONDARY_L": secondary_l,
            "SECONDARY_O": secondary_o,
        }
        if secondary_order
        else {}
    )
    bins = no if site_routing else tr.cdiv(no, bo)
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
        bo,
        256,
        OUTPUT=output,
        SITE_ROUTING=site_routing,
        **routing_options,
        num_warps=4,
        enable_fp_fusion=False,
    )
    sorted_keys, order = torch.sort(keys)
    thresholds = torch.arange(bins + 1, device=packed.device, dtype=torch.int32)
    if secondary_order:
        # Search composite-key bin starts directly: no A-sized primary-key copy.
        thresholds.mul_(secondary_n)
    boundaries = torch.searchsorted(sorted_keys, thresholds)
    max_distance = distances.amax()
    return order, boundaries, max_distance


def uses_input_site_routing(recipe):
    from .recipe import InputSiteStreamingHRecipe

    return type(recipe) is InputSiteStreamingHRecipe


def uses_input_secondary_order(recipe):
    from .recipe import InputOrderStreamingHRecipe

    return type(recipe) is InputOrderStreamingHRecipe


def uses_site_routing(recipe):
    from .recipe import (
        InputOrderStreamingHRecipe,
        InputSiteStreamingHRecipe,
        OwnerBatchHRecipe,
        OwnerBatchStreamingHRecipe,
        SiteRoutedHRecipe,
        SiteRoutedStreamingHRecipe,
    )

    return type(recipe) in (
        SiteRoutedHRecipe,
        SiteRoutedStreamingHRecipe,
        OwnerBatchHRecipe,
        OwnerBatchStreamingHRecipe,
        InputSiteStreamingHRecipe,
        InputOrderStreamingHRecipe,
    )


def h_capacity(recipe):
    from .recipe import (
        GroupedOutputHRecipe,
        InputOrderStreamingHRecipe,
        InputOwnedHRecipe,
        InputSiteStreamingHRecipe,
        OwnerBatchHRecipe,
        OwnerBatchStreamingHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        SiteRoutedHRecipe,
        SiteRoutedStreamingHRecipe,
        StreamingInputHRecipe,
    )

    return (
        recipe.h_batch
        if type(recipe)
        in (
            GroupedOutputHRecipe,
            InputOwnedHRecipe,
            SiteRoutedHRecipe,
            OwnerBatchHRecipe,
            SiteRoutedStreamingHRecipe,
            OwnerBatchStreamingHRecipe,
            InputSiteStreamingHRecipe,
            InputOrderStreamingHRecipe,
            StreamingInputHRecipe,
            ParallelReusedHRecipe,
            PreparedReusedHRecipe,
        )
        else recipe.batch_tile
    )


def allocate_h(x, atom_count, recipe):
    from .recipe import (
        GroupedOutputHRecipe,
        InputOrderStreamingHRecipe,
        InputOwnedHRecipe,
        InputSiteStreamingHRecipe,
        OwnerBatchHRecipe,
        OwnerBatchStreamingHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        SiteRoutedHRecipe,
        SiteRoutedStreamingHRecipe,
        StreamingInputHRecipe,
    )

    if type(recipe) in (
        GroupedOutputHRecipe,
        InputOwnedHRecipe,
        SiteRoutedHRecipe,
        OwnerBatchHRecipe,
        SiteRoutedStreamingHRecipe,
        OwnerBatchStreamingHRecipe,
        InputSiteStreamingHRecipe,
        InputOrderStreamingHRecipe,
        StreamingInputHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
    ):
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


def prepare_output_fields(packed, routing, *, output=True):
    """Only the three output fields survive across H chunks, never backward."""
    import triton as tr

    from . import output_kernels as kernels

    a = packed.shape[1]
    hot = packed.new_empty((3, a))
    unsafe = torch.zeros((), device=packed.device, dtype=torch.int32)
    kernels.prepare_output_fields[(tr.cdiv(a, 256),)](
        packed,
        routing[0],
        hot,
        unsafe,
        a,
        256,
        OUTPUT=output,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return hot, unsafe


def owner_batch_tile(recipe):
    return getattr(recipe, "owner_batch_tile", recipe.batch_tile)


def active_owner_tiles(h, batch, batch_start, recipe):
    capacity = (h.shape[0] if h.ndim == 3 else 1) * recipe.batch_tile
    return (
        min(capacity, batch - batch_start) + owner_batch_tile(recipe) - 1
    ) // owner_batch_tile(recipe)


def aggregate_chunk(
    x, packed, sizes, recipe, routing, y, *, h=None, batch_start=0, hot=None
):
    import triton as tr

    from . import output_kernels as kernels

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    tiles = (
        tr.cdiv(b, owner_batch_tile(recipe))
        if h is None
        else active_owner_tiles(h, b, batch_start, recipe)
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
        owner_batch_tile(recipe),
        recipe.patch_sites,
        getattr(recipe, "output_group", recipe.atom_group),
        recipe.output_tile,
    )
    options = {
        "H": h,
        "BSTART": batch_start,
        "SITE_ROUTING": uses_site_routing(recipe),
        "H_BM": recipe.batch_tile if hasattr(recipe, "owner_batch_tile") else 0,
        "num_warps": 4,
        "enable_fp_fusion": False,
    }
    grid = (tr.cdiv(no, recipe.output_tile), tiles)
    if hot is None:
        kernels.output_owned[grid](*args, CACHED=h is not None, **options)
    else:
        if h is None:
            raise ValueError("prepared output ownership requires a cached H chunk")
        for fallback in (False, True):
            fallback_args = (
                (*args[:-2], recipe.atom_group, args[-1]) if fallback else args
            )
            kernels.prepared_output_owned[grid](
                *fallback_args, Hot=hot[0], Unsafe=hot[1], FALLBACK=fallback, **options
            )


def forward_output_owned(x, packed, sizes, recipe):
    from .recipe import (
        GroupedOutputHRecipe,
        InputOrderStreamingHRecipe,
        InputOwnedHRecipe,
        InputSiteStreamingHRecipe,
        OwnerBatchHRecipe,
        OwnerBatchStreamingHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        ReusedHRecipe,
        SiteRoutedHRecipe,
        SiteRoutedStreamingHRecipe,
        StreamingInputHRecipe,
    )

    b, _, no, *_ = sizes
    a = packed.shape[1]
    if not a:
        return x.new_zeros((b, no))
    y = x.new_empty((b, no))
    routing = prepare_routing(packed, sizes, recipe)
    hot = (
        prepare_output_fields(packed, routing)
        if type(recipe)
        in (
            GroupedOutputHRecipe,
            InputOwnedHRecipe,
            SiteRoutedHRecipe,
            OwnerBatchHRecipe,
            SiteRoutedStreamingHRecipe,
            OwnerBatchStreamingHRecipe,
            InputSiteStreamingHRecipe,
            InputOrderStreamingHRecipe,
            StreamingInputHRecipe,
            PreparedReusedHRecipe,
        )
        else None
    )
    if type(recipe) in (
        GroupedOutputHRecipe,
        InputOwnedHRecipe,
        SiteRoutedHRecipe,
        OwnerBatchHRecipe,
        SiteRoutedStreamingHRecipe,
        OwnerBatchStreamingHRecipe,
        InputSiteStreamingHRecipe,
        InputOrderStreamingHRecipe,
        StreamingInputHRecipe,
        ReusedHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
    ):
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
