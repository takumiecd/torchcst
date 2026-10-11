"""Saved one-entry dyadic support index; bounded H/G and owner partials."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .tile_dot_executor import allocate_scratch


def levels(n, tile):
    bins = (n + tile - 1) // tile
    return (bins - 1).bit_length() + 1


def prepare_tiles(packed, sizes, recipe, *, out=None):
    import triton as tr

    from . import tile_bucket_kernels as k

    a = packed.shape[1]
    routes = []
    for output in (False, True):
        n = sizes[2] if output else sizes[1]
        bins = tr.cdiv(n, recipe.tile_sites)
        count = levels(n, recipe.tile_sites)
        keys = torch.empty(a, device=packed.device, dtype=torch.int32)
        k.keys[(tr.cdiv(a, 256),)](
            packed,
            keys,
            a,
            n,
            recipe.tile_sites,
            output,
            count,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        sorted_keys, order64 = torch.sort(keys)
        if out is None:
            order = order64.to(torch.int32)
        else:
            order = out[2 * output]
            order.copy_(order64)
        bounds = torch.searchsorted(
            sorted_keys,
            torch.arange(count * bins + 1, device=packed.device, dtype=torch.int32),
            out_int32=True,
            out=None if out is None else out[2 * output + 1],
        )
        routes.extend((order, bounds))
    return tuple(routes)


def produce(
    x,
    packed,
    routing,
    sizes,
    recipe,
    h,
    start,
    *,
    output=False,
    derivative=False,
    consume=0,
    partner=None,
    physical=None,
):
    import triton as tr

    from . import tile_bucket_kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    n, length, origin = (no, lo, oo) if output else (ni, li, oi)
    k.produce[(tr.cdiv(a, 32),)](
        x,
        packed,
        routing[2 if output else 0],
        h,
        partner,
        physical,
        a,
        b,
        n,
        length,
        origin,
        *x.stride(),
        start,
        recipe.batch_tile,
        recipe.tile_sites,
        output,
        derivative,
        consume,
        levels(n, recipe.tile_sites),
        num_warps=4,
        enable_fp_fusion=False,
    )


def aggregate(h, packed, routing, sizes, recipe, scratch, result, start, *, output):
    import triton as tr

    from . import tile_bucket_kernels as k
    from . import tile_dot_kernels

    b, ni, no, li, lo, oi, oo = sizes
    n, length, origin = (no, lo, oo) if output else (ni, li, oi)
    offset = 2 if output else 0
    k.owner[(tr.cdiv(n, recipe.tile_sites), recipe.owner_parts)](
        h,
        packed,
        routing[offset],
        routing[offset + 1],
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


class _TileBucket(torch.autograd.Function):
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
            routing = (torch.empty(0, device=x.device, dtype=torch.int32),) * 4
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


def tile_bucket_h(x, p, value, sizes, recipe):
    return _TileBucket.apply(x, p, value, sizes, recipe)
