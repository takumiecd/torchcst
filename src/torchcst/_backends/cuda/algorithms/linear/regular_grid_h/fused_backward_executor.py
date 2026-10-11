"""Matched forward and snapshots; G ends inside each backward input owner."""

from dataclasses import asdict

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare
from .recipe import InputBatchProducerHRecipe


def forward_recipe(recipe):
    settings = asdict(recipe)
    settings.pop("backward_batch_tile")
    return InputBatchProducerHRecipe(**settings)


class _FusedBackwardH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        from .executor import forward_contraction

        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        matched = forward_recipe(recipe)
        packed = _prepare(source, value, sizes, matched)
        y = forward_contraction(x, packed, sizes, matched)
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(x, source, amplitude_max, packed)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import fused_backward_kernels, kernels
        from .output_owner import prepare_routing

        x, source, amplitude_max, packed = ctx.saved_tensors
        b, ni = ctx.sizes[:2]
        recipe = ctx.recipe
        need_x, need_p = ctx.needs_input_grad[:2]
        a, bm = len(source), recipe.backward_batch_tile
        tiles = tr.cdiv(b, bm)
        dx = x.new_empty(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        # Unvisited empty input supports and padded rows are exact zeros. Only
        # one spatial owner writes each live atom/batch record; no grad atomics.
        partial = x.new_zeros((tiles, 3, a)) if need_p else None
        if a and b and (need_x or need_p):
            if need_x:
                matched = forward_recipe(recipe)
                routing = prepare_routing(
                    packed, ctx.sizes, matched, output=False, tile=recipe.input_tile
                )
                fused_backward_kernels.fused_input_owned[
                    (tr.cdiv(ni, recipe.input_tile), tiles)
                ](
                    x,
                    dy,
                    packed,
                    *routing,
                    dx,
                    partial,
                    a,
                    *ctx.sizes,
                    *x.stride(),
                    *dy.stride(),
                    need_p,
                    bm,
                    recipe.patch_sites,
                    recipe.atom_group,
                    recipe.input_tile,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                kernels.backward[(tr.cdiv(a, recipe.atom_group), tiles)](
                    x,
                    dy,
                    packed,
                    None,
                    partial,
                    a,
                    *ctx.sizes,
                    *x.stride(),
                    *dy.stride(),
                    False,
                    True,
                    bm,
                    recipe.patch_sites,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        elif need_x:
            dx.zero_()
        if a and need_p:
            kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
                partial,
                dp,
                source,
                amplitude_max,
                a,
                tiles,
                tr.next_power_of_2(max(tiles, 1)),
                recipe.prep_group,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def fused_backward_h(x, p, value, sizes, recipe):
    return _FusedBackwardH.apply(x, p, value, sizes, recipe)
