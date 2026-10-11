"""Two separately selectable tile-dot experiments; never combine effects."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .tile_dot_executor import aggregate, allocate_scratch, prepare_tiles
from .tile_dot_executor import produce as fixed_produce


def produce(
    x,
    packed,
    routing,
    sizes,
    recipe,
    h,
    dh,
    start,
    *,
    output=False,
    derivative=False,
    window=False,
    consume=0,
    partner=None,
    physical=None,
):
    import triton as tr

    from . import tile_dot_kernels as fixed
    from . import tile_schedule_kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    offset = 3 if output else 0
    args = (
        a,
        b,
        no if output else ni,
        lo if output else li,
        oo if output else oi,
        *x.stride(),
        start,
        recipe.batch_tile,
    )
    k.produce[(tr.cdiv(2 * a, 32),)](
        x,
        packed,
        routing[offset],
        routing[offset + 2],
        h,
        dh,
        partner,
        physical,
        *args,
        recipe.tile_sites,
        output,
        derivative,
        window,
        consume,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if consume:
        k.fallback_consume[(tr.cdiv(a, 32),)](
            x,
            packed,
            routing[offset + 2],
            h,
            partner,
            physical,
            *args,
            output,
            consume,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        fixed.fallback_produce[(tr.cdiv(a, 32),)](
            x,
            packed,
            routing[offset + 2],
            h,
            dh,
            *args,
            recipe.tile_sites,
            output,
            derivative,
            num_warps=4,
            enable_fp_fusion=False,
        )


class _TileSchedule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe, window):
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe)
        ctx.sizes, ctx.recipe, ctx.window = sizes, recipe, window
        a, b = len(source), sizes[0]
        y = x.new_empty((b, sizes[2]))
        if a and b:
            routing = prepare_tiles(packed, sizes, recipe)
            h = x.new_empty((a, recipe.batch_tile))
            scratch = allocate_scratch(x, sizes, recipe)
            for start in range(0, b, recipe.batch_tile):
                if window:
                    produce(
                        x, packed, routing, sizes, recipe, h, None, start, window=True
                    )
                else:
                    fixed_produce(x, packed, routing, sizes, recipe, h, None, start)
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

        from . import kernels, tile_dot_kernels

        x, source, amplitude_max, packed, *routing = ctx.saved_tensors
        sizes, recipe, window = ctx.sizes, ctx.recipe, ctx.window
        a, b = len(source), sizes[0]
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_empty(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        physical = x.new_zeros((3, a)) if need_p else None
        if a and b and (need_x or need_p):
            g = x.new_empty((a, recipe.batch_tile))
            h = torch.empty_like(g) if need_p else None
            if need_p and window:
                dh, dg = torch.empty_like(g), torch.empty_like(g)
            else:
                dh = dg = None
            scratch = allocate_scratch(x, sizes, recipe) if need_x else None
            for start in range(0, b, recipe.batch_tile):
                if window:
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
                        window=True,
                    )
                    if need_p:
                        produce(
                            x,
                            packed,
                            routing,
                            sizes,
                            recipe,
                            h,
                            dh,
                            start,
                            derivative=True,
                            window=True,
                        )
                        tile_dot_kernels.physical[(tr.cdiv(a, 64),)](
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
                elif need_p:
                    # H is contracted first to consume dG inside its producer.
                    # Recompute H/dH later to consume dH against G. Only H/G
                    # value slabs are allocated; neither derivative has a slab.
                    fixed_produce(x, packed, routing, sizes, recipe, h, None, start)
                    produce(
                        dy,
                        packed,
                        routing,
                        sizes,
                        recipe,
                        g,
                        None,
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
                        None,
                        start,
                        derivative=True,
                        consume=2,
                        partner=g,
                        physical=physical,
                    )
                else:
                    fixed_produce(
                        dy, packed, routing, sizes, recipe, g, None, start, output=True
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
        return dx, dp, None, None, None, None


def tile_schedule_h(x, p, value, sizes, recipe, *, window):
    return _TileSchedule.apply(x, p, value, sizes, recipe, window)
