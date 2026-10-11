"""Small-batch H/G slabs, retained support routing, no full W or B x A H."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare


def prepare_tiles(packed, sizes, recipe):
    import triton as tr

    from . import tile_dot_kernels as k

    a = packed.shape[1]
    routes = []
    for output in (False, True):
        n = sizes[2] if output else sizes[1]
        keys = torch.empty(2 * a, device=packed.device, dtype=torch.int32)
        wide = torch.empty(tr.cdiv(a, 256), device=packed.device, dtype=torch.int32)
        k.membership[(tr.cdiv(a, 256),)](
            packed,
            keys,
            wide,
            a,
            n,
            recipe.tile_sites,
            output,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        sorted_keys, order = torch.sort(keys)
        bounds = torch.searchsorted(
            sorted_keys,
            torch.arange(
                tr.cdiv(n, recipe.tile_sites) + 1,
                device=packed.device,
                dtype=torch.int32,
            ),
        )
        routes.extend((order, bounds, wide.amax().reshape(1)))
    return tuple(routes)


def produce(
    x, packed, routing, sizes, recipe, h, dh, start, *, output=False, derivative=False
):
    import triton as tr

    from . import tile_dot_kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    offset = 3 if output else 0
    k.produce[(tr.cdiv(2 * a, 32),)](
        x,
        packed,
        routing[offset],
        routing[offset + 2],
        h,
        dh,
        a,
        b,
        no if output else ni,
        lo if output else li,
        oo if output else oi,
        *x.stride(),
        start,
        recipe.batch_tile,
        recipe.tile_sites,
        output,
        derivative,
        num_warps=4,
        enable_fp_fusion=False,
    )


def aggregate(h, packed, routing, sizes, recipe, scratch, result, start, *, output):
    import triton as tr

    from . import tile_dot_kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    n, length, origin = (no, lo, oo) if output else (ni, li, oi)
    a = packed.shape[1]
    offset = 3 if output else 0
    k.owner[(tr.cdiv(n, recipe.tile_sites), recipe.owner_parts)](
        h,
        packed,
        routing[offset],
        routing[offset + 1],
        routing[offset + 2],
        scratch,
        a,
        b,
        n,
        length,
        origin,
        start,
        recipe.batch_tile,
        recipe.tile_sites,
        recipe.owner_parts,
        output,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.reduce_owner[(tr.cdiv(n, recipe.tile_sites),)](
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


def allocate_scratch(x, sizes, recipe):
    import triton as tr

    return x.new_empty(
        (
            tr.cdiv(max(sizes[1:3]), recipe.tile_sites),
            recipe.owner_parts,
            recipe.batch_tile,
            recipe.tile_sites,
        )
    )


class _TileDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe)
        ctx.sizes, ctx.recipe = sizes, recipe
        a, b = len(source), sizes[0]
        y = x.new_empty((b, sizes[2]))
        if a and b:
            routing = prepare_tiles(packed, sizes, recipe)
            h = x.new_empty((a, recipe.batch_tile))
            scratch = allocate_scratch(x, sizes, recipe)
            for start in range(0, b, recipe.batch_tile):
                produce(x, packed, routing, sizes, recipe, h, None, start)
                aggregate(
                    h, packed, routing, sizes, recipe, scratch, y, start, output=True
                )
        else:
            y.zero_()
            routing = (torch.empty(0, device=x.device, dtype=torch.int64),) * 6
        ctx.save_for_backward(x, source, amplitude_max, packed, *routing)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels
        from . import tile_dot_kernels as k

        x, source, amplitude_max, packed, *routing = ctx.saved_tensors
        sizes, recipe = ctx.sizes, ctx.recipe
        a, b = len(source), sizes[0]
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_empty(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        physical = x.new_zeros((3, a)) if need_p else None
        if a and b and (need_x or need_p):
            g = x.new_empty((a, recipe.batch_tile))
            if need_p:
                h, dh, dg = (torch.empty_like(g) for _ in range(3))
            else:
                h = dh = dg = None
            scratch = allocate_scratch(x, sizes, recipe) if need_x else None
            for start in range(0, b, recipe.batch_tile):
                produce(
                    dy,
                    packed,
                    routing,
                    sizes,
                    recipe,
                    g,
                    dg,
                    start,
                    output=True,
                    derivative=need_p,
                )
                if need_p:
                    produce(
                        x, packed, routing, sizes, recipe, h, dh, start, derivative=True
                    )
                    k.physical[(tr.cdiv(a, 64),)](
                        h,
                        dh,
                        g,
                        dg,
                        packed,
                        physical,
                        a,
                        b,
                        start,
                        recipe.batch_tile,
                        64,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
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


def tile_dot_h(x, p, value, sizes, recipe):
    return _TileDot.apply(x, p, value, sizes, recipe)
