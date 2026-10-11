"""Controlled workspace lifetime; identical normalized owner contractions."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .tile_bucket_executor import produce
from .tile_dot_executor import allocate_scratch
from .tile_owner_prep_executor import aggregate, prepare_tiles


def allocate_arena(x, p, sizes, recipe):
    """Disjoint aligned views, owned by one forward, with no copied intermediates."""
    a = len(p)
    shapes = [
        (a, 4),
        (13, a),
        (a, recipe.batch_tile),
        (a, recipe.batch_tile),
        (3, a),
        ((max(sizes[1:3]) + 15) // 16, 8, 16, 16),
    ]
    dtypes = [x.dtype] * len(shapes)
    for n in sizes[1:3]:
        bins = (n + 15) // 16
        count = (bins - 1).bit_length() + 1
        shapes.extend([(a,), (count * bins + 1,)])
        dtypes.extend([torch.int32, torch.int32])
    for n in sizes[1:3]:
        bins = (n + 15) // 16
        count = (bins - 1).bit_length() + 1
        shapes.extend([(count + 1,), (bins, 8, count, 2, 2), (2, a)])
        dtypes.extend([torch.int32] * 3)
    import math

    lengths = [math.prod(shape) for shape in shapes]
    padded = [(length + 127) // 128 * 128 for length in lengths]
    arena = x.new_empty(sum(padded))
    buffers = []
    offset = 0
    for shape, dtype, length, span in zip(shapes, dtypes, lengths, padded, strict=True):
        buffers.append(arena[offset : offset + length].view(dtype).reshape(shape))
        offset += span
    return buffers


class _TileOwnerLifetime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        arena = recipe.lifetime == "arena"
        buffers = allocate_arena(x, p, sizes, recipe) if arena else None
        source = (
            buffers[0].copy_(p)
            if arena
            else p.clone(memory_format=torch.contiguous_format)
        )
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(
            source,
            value,
            sizes,
            recipe,
            canonical_centers=True,
            out=buffers[1] if arena else None,
        )
        ctx.sizes, ctx.recipe = sizes, recipe
        a, b = len(source), sizes[0]
        y = x.new_empty((b, sizes[2]))
        reuse = recipe.lifetime in ("reuse", "combined", "arena")
        empty = x.new_empty(0)
        saved_h = saved_scratch = saved_g = saved_physical = empty
        if a and b:
            routing = prepare_tiles(
                packed, sizes, recipe, out=buffers[6:] if arena else None
            )
            h = buffers[2] if arena else x.new_empty((a, recipe.batch_tile))
            scratch = buffers[5] if arena else allocate_scratch(x, sizes, recipe)
            if reuse:
                if ctx.needs_input_grad[1]:
                    saved_h = h
                if ctx.needs_input_grad[0]:
                    saved_scratch = scratch
            if arena:
                saved_g, saved_physical = buffers[3:5]
            for start in range(0, b, recipe.batch_tile):
                produce(x, packed, routing, sizes, recipe, h, start)
                aggregate(
                    h,
                    packed,
                    routing,
                    sizes,
                    recipe,
                    scratch,
                    y,
                    start,
                    output=True,
                    direct=x,
                )
        else:
            y.zero_()
            routing = (torch.empty(0, device=x.device, dtype=torch.int32),) * 10
        ctx.save_for_backward(
            x,
            source,
            amplitude_max,
            packed,
            *routing,
            saved_h,
            saved_scratch,
            saved_g,
            saved_physical,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels

        x, source, amplitude_max, packed, *saved = ctx.saved_tensors
        routing = saved[:-4]
        saved_h, saved_scratch, saved_g, saved_physical = saved[-4:]
        sizes, recipe = ctx.sizes, ctx.recipe
        arena = recipe.lifetime == "arena"
        a, b = len(source), sizes[0]
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_empty(x.shape) if need_x else None
        reuse = recipe.lifetime in ("reuse", "combined", "arena")
        late = recipe.lifetime in ("late", "combined", "arena")
        dp = torch.empty_like(source) if need_p and not late else None
        g = h = scratch = None
        if need_p and arena and a and b:
            from .tile_owner_lifetime_kernels import clear

            physical = saved_physical
            clear[(tr.cdiv(physical.numel(), 512),)](
                physical, physical.numel(), 512, num_warps=4
            )
        else:
            physical = x.new_zeros((3, a)) if need_p else None
        if a and b and (need_x or need_p):
            g = saved_g if arena else x.new_empty((a, recipe.batch_tile))
            h = (saved_h if reuse else torch.empty_like(g)) if need_p else None
            scratch = (
                (saved_scratch if reuse else allocate_scratch(x, sizes, recipe))
                if need_x
                else None
            )
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
                        direct=x,
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
                        direct=x,
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
                        direct=dy,
                    )
        elif need_x:
            dx.zero_()
        # dP does not overlap the producer slabs in the delayed variant.
        # Saved workspaces belong to this forward and are released by autograd.
        del g, h, scratch
        if need_p and late:
            dp = torch.empty_like(source)
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


def tile_owner_lifetime_h(x, p, value, sizes, recipe):
    return _TileOwnerLifetime.apply(x, p, value, sizes, recipe)
