"""Canonical packed/partial positions, mapped directly to raw IDs in the VJP."""

from dataclasses import asdict

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .recipe import InputBatchProducerHRecipe


def prepare_canonical(source, value, sizes, recipe):
    """Sort small keys first, write the sole packed buffer in final order."""
    import triton as tr

    from . import canonical_kernels

    _, ni, _, li, _, oi, _ = sizes
    a = len(source)
    keys = torch.empty((a,), device=source.device, dtype=torch.int32)
    if a:
        canonical_kernels.source_input_keys[(tr.cdiv(a, 256),)](
            source, keys, a, ni, li, oi, 256, num_warps=4, enable_fp_fusion=False
        )
    sorted_keys, raw_order = torch.sort(keys)
    packed = _prepare(source, value, sizes, recipe, order=raw_order)
    thresholds = torch.arange(ni + 1, device=source.device, dtype=torch.int32)
    bounds = torch.searchsorted(sorted_keys, thresholds)
    if a:
        distances = torch.empty(
            (tr.cdiv(a, 256),), device=source.device, dtype=torch.int32
        )
        canonical_kernels.input_distances[(tr.cdiv(a, 256),)](
            packed, sorted_keys, distances, a, 256, num_warps=4, enable_fp_fusion=False
        )
        max_distance = distances.amax()
    else:
        max_distance = torch.zeros((), device=source.device, dtype=torch.int32)
    return packed, raw_order, bounds, max_distance


class _CanonicalH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        from .output_owner import forward_output_owned

        # Common owner scheduling uses the existing BM16/G16 recipe. Parameter
        # storage and optimizer state remain in raw order throughout.
        schedule = InputBatchProducerHRecipe(**asdict(recipe))
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed, order, bounds, distance = prepare_canonical(
            source, value, sizes, schedule
        )
        y = forward_output_owned(x, packed, sizes, schedule)
        ctx.sizes, ctx.recipe = sizes, schedule
        ctx.save_for_backward(x, source, amplitude_max, packed, order, bounds, distance)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels
        from .input_owner import input_owned_backward

        x, source, amplitude_max, packed, order, bounds, distance = ctx.saved_tensors
        sizes, recipe = ctx.sizes, ctx.recipe
        b, ni, no, li, lo, oi, oo = sizes
        a = len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_zeros(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        tiles = tr.cdiv(b, recipe.batch_tile)
        partial_tiles = (
            tr.cdiv(min(b, recipe.g_batch), recipe.producer_batch_tile)
            if need_x
            else tiles
        )
        partial = x.new_empty((partial_tiles, 3, a)) if need_p else None
        physical = x.new_empty((3, a)) if need_x and need_p else None
        if a and (need_x or need_p):
            if need_x:
                input_owned_backward(
                    x,
                    dy,
                    packed,
                    sizes,
                    recipe,
                    dx,
                    partial,
                    need_p,
                    physical=physical,
                    routing=(None, bounds, distance),
                    identity_order=True,
                )
            else:
                kernels.backward[(tr.cdiv(a, recipe.atom_group), tiles)](
                    x,
                    dy,
                    packed,
                    dx,
                    partial,
                    a,
                    b,
                    ni,
                    no,
                    li,
                    lo,
                    oi,
                    oo,
                    *x.stride(),
                    *dy.stride(),
                    need_x,
                    need_p,
                    recipe.batch_tile,
                    recipe.patch_sites,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            if need_p:
                if physical is not None:
                    partial, tiles = physical[None], 1
                kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
                    partial,
                    dp,
                    source,
                    amplitude_max,
                    a,
                    tiles,
                    tr.next_power_of_2(tiles),
                    recipe.prep_group,
                    Order=order,
                    RESTORE_ORDER=True,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None


def canonical_h(x, p, value, sizes, recipe):
    return _CanonicalH.apply(x, p, value, sizes, recipe)
