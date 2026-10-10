"""Backward-only routing and bounded G, rebuilt from the forward snapshot."""


def input_owned_backward(
    x, dy, packed, sizes, recipe, dx, partial, need_p, *, physical=None
):
    import triton as tr

    from . import input_kernels, output_kernels
    from .output_owner import (
        active_h_tiles,
        allocate_h,
        h_capacity,
        prepare_output_fields,
        prepare_routing,
    )

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    routing = prepare_routing(
        packed, sizes, recipe, output=False, tile=recipe.input_tile
    )
    hot = prepare_output_fields(packed, routing, output=False)
    streamed = hasattr(recipe, "g_batch")
    capacity = recipe.g_batch if streamed else h_capacity(recipe)
    g = (
        x.new_empty((capacity // recipe.batch_tile, a, recipe.batch_tile))
        if streamed
        else allocate_h(x, a, recipe)
    )
    if physical is not None and not b:
        physical.zero_()
    # Symmetric owner schedule: output sites of this contraction are input sites.
    swapped = (b, no, ni, lo, li, oo, oi)
    args = (
        dy,
        packed,
        *routing,
        dx,
        a,
        *swapped,
        *dy.stride(),
        recipe.batch_tile,
        recipe.patch_sites,
    )
    for start in range(0, b, capacity):
        tiles = active_h_tiles(g, b, start, recipe.batch_tile)
        input_kernels.produce_g_parameters[(tr.cdiv(a, recipe.atom_group), tiles)](
            x,
            dy,
            packed,
            routing[0],
            g,
            partial,
            a,
            *sizes,
            *x.stride(),
            *dy.stride(),
            need_p,
            start,
            recipe.batch_tile,
            recipe.patch_sites,
            recipe.atom_group,
            STREAM_PARTIAL=streamed,
            num_warps=4,
            enable_fp_fusion=False,
        )
        for fallback in (False, True):
            group = recipe.atom_group if fallback else recipe.output_group
            output_kernels.prepared_output_owned[
                (tr.cdiv(ni, recipe.input_tile), tiles)
            ](
                *args,
                group,
                recipe.input_tile,
                H=g,
                Hot=hot[0],
                Unsafe=hot[1],
                BSTART=start,
                FALLBACK=fallback,
                PROFILE_OUTPUT=False,
                num_warps=4,
                enable_fp_fusion=False,
            )

        if physical is not None:
            input_kernels.accumulate_physical[(tr.cdiv(a, recipe.prep_group),)](
                partial,
                physical,
                a,
                tiles,
                tr.next_power_of_2(g.shape[0]),
                recipe.prep_group,
                start == 0,
                num_warps=4,
                enable_fp_fusion=False,
            )
