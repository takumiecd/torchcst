"""Controlled owner metadata snapshots; unchanged bucket H/G producers."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .tile_bucket_executor import levels, produce
from .tile_bucket_executor import prepare_tiles as bucket_tiles
from .tile_dot_executor import allocate_scratch


def prepare_bands(bounds, bins, count, *, out=None):
    import triton as tr

    from . import tile_owner_prep_kernels as k

    result = bounds.new_zeros(count + 1) if out is None else out.zero_()
    k.prepare_bands[(1,)](
        bounds, result, bins, count, tr.next_power_of_2(count), num_warps=4
    )
    return result


def prepare_ranges(bounds, n, recipe, count, *, out=None):
    import triton as tr

    from . import tile_owner_prep_kernels as k

    bins = tr.cdiv(n, recipe.tile_sites)
    result = (
        bounds.new_empty((bins, recipe.owner_parts, count, 2, 2))
        if out is None
        else out
    )
    k.prepare_ranges[(tr.cdiv(result.numel() // 2, 256),)](
        bounds,
        result,
        n,
        recipe.tile_sites,
        recipe.owner_parts,
        count,
        256,
        num_warps=4,
    )
    return result


def prepare_membership(packed, n, recipe, output, *, out=None):
    import triton as tr

    from . import tile_owner_prep_kernels as k

    a = packed.shape[1]
    result = (
        torch.empty((2, a), device=packed.device, dtype=torch.int32)
        if out is None
        else out
    )
    k.prepare_membership[(tr.cdiv(a, 256),)](
        packed, result, a, n, recipe.tile_sites, output, 256, num_warps=4
    )
    return result


def prepare_tiles(packed, sizes, recipe, *, out=None):
    routing = bucket_tiles(packed, sizes, recipe, out=None if out is None else out[:4])
    extra = []
    empty = routing[0].new_empty(0)
    for output in (False, True):
        n = sizes[2] if output else sizes[1]
        count = levels(n, recipe.tile_sites)
        bounds = routing[2 * output + 1]
        extra.extend(
            (
                prepare_bands(
                    bounds,
                    (n + recipe.tile_sites - 1) // recipe.tile_sites,
                    count,
                    out=None if out is None else out[4 + 3 * output],
                )
                if recipe.mode in ("bands", "all")
                else empty,
                prepare_ranges(
                    bounds,
                    n,
                    recipe,
                    count,
                    out=None if out is None else out[5 + 3 * output],
                )
                if recipe.mode in ("ranges", "all")
                else empty,
                prepare_membership(
                    packed,
                    n,
                    recipe,
                    output,
                    out=None if out is None else out[6 + 3 * output],
                )
                if recipe.mode in ("membership", "all")
                else empty,
            )
        )
    return (*routing, *extra)


def aggregate(h, packed, routing, sizes, recipe, scratch, result, start, *, output):
    import triton as tr

    from . import tile_dot_kernels
    from . import tile_owner_prep_kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    n, length, origin = (no, lo, oo) if output else (ni, li, oi)
    offset = 2 if output else 0
    k.owner[(tr.cdiv(n, recipe.tile_sites), recipe.owner_parts)](
        h,
        packed,
        routing[offset],
        routing[offset + 1],
        *routing[4 + 3 * output : 7 + 3 * output],
        scratch,
        packed.shape[1],
        b,
        n,
        length,
        origin,
        start,
        recipe.batch_tile,
        recipe.tile_sites,
        recipe.owner_parts,
        output,
        levels(n, recipe.tile_sites),
        recipe.mode in ("bands", "all"),
        recipe.mode in ("ranges", "all"),
        recipe.mode in ("membership", "all"),
        num_warps=4,
        enable_fp_fusion=False,
    )
    tile_dot_kernels.reduce_owner[(tr.cdiv(n, recipe.tile_sites),)](
        scratch,
        result,
        b,
        n,
        start,
        recipe.batch_tile,
        recipe.tile_sites,
        recipe.owner_parts,
        num_warps=4,
        enable_fp_fusion=False,
    )


class _TileOwnerPrep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe, canonical_centers=True)
        ctx.sizes, ctx.recipe = sizes, recipe
        a, b = len(source), sizes[0]
        y = x.new_empty((b, sizes[2]))
        if a and b:
            routing = prepare_tiles(packed, sizes, recipe)
            h = x.new_empty((a, recipe.batch_tile))
            scratch = allocate_scratch(x, sizes, recipe)
            for start in range(0, b, recipe.batch_tile):
                produce(x, packed, routing, sizes, recipe, h, start)
                aggregate(
                    h, packed, routing, sizes, recipe, scratch, y, start, output=True
                )
        else:
            y.zero_()
            routing = (torch.empty(0, device=x.device, dtype=torch.int32),) * 10
        ctx.save_for_backward(x, source, amplitude_max, packed, *routing)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels

        x, source, amplitude_max, packed, *routing = ctx.saved_tensors
        sizes, recipe = ctx.sizes, ctx.recipe
        a, b = len(source), sizes[0]
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_empty(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        physical = x.new_zeros((3, a)) if need_p else None
        if a and b and (need_x or need_p):
            g = x.new_empty((a, recipe.batch_tile))
            h = torch.empty_like(g) if need_p else None
            scratch = allocate_scratch(x, sizes, recipe) if need_x else None
            for start in range(0, b, recipe.batch_tile):
                if need_p:
                    produce(x, packed, routing, sizes, recipe, h, start)
                    produce(
                        dy,
                        packed,
                        routing,
                        sizes,
                        recipe,
                        g,
                        start,
                        output=True,
                        derivative=True,
                        consume=1,
                        partner=h,
                        physical=physical,
                    )
                    produce(
                        x,
                        packed,
                        routing,
                        sizes,
                        recipe,
                        h,
                        start,
                        derivative=True,
                        consume=2,
                        partner=g,
                        physical=physical,
                    )
                else:
                    produce(dy, packed, routing, sizes, recipe, g, start, output=True)
                if need_x:
                    aggregate(
                        g,
                        packed,
                        routing,
                        sizes,
                        recipe,
                        scratch,
                        dx,
                        start,
                        output=False,
                    )
        elif need_x:
            dx.zero_()
        if a and need_p:
            kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
                physical[None],
                dp,
                source,
                amplitude_max,
                a,
                1,
                1,
                recipe.prep_group,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def tile_owner_prep_h(x, p, value, sizes, recipe):
    return _TileOwnerPrep.apply(x, p, value, sizes, recipe)
