"""Snapshot/preparation once; H/G live only inside each atom/batch CTA."""

import torch
from torch.autograd.function import once_differentiable

from ..periodic_product.executor import _prepare


def forward_contraction(x, packed, sizes, recipe):
    import triton as tr

    from . import kernels
    from .recipe import (
        OutputOwnedHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        ReusedHRecipe,
    )

    if type(recipe) in (
        OutputOwnedHRecipe,
        ParallelReusedHRecipe,
        PreparedReusedHRecipe,
        ReusedHRecipe,
    ):
        from .output_owner import forward_output_owned

        return forward_output_owned(x, packed, sizes, recipe)

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    y = x.new_zeros((b, no))
    if a:
        kernels.forward[(tr.cdiv(a, recipe.atom_group), tr.cdiv(b, recipe.batch_tile))](
            x,
            packed,
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
            num_warps=4,
            enable_fp_fusion=False,
        )
    return y


class _OnchipH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        source = p.clone(memory_format=torch.contiguous_format)
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe)
        y = forward_contraction(x, packed, sizes, recipe)
        ctx.sizes, ctx.recipe = sizes, recipe
        # Prepared values include forward-time width, norms and centre VJPs.
        # Geometry/scalar placement is captured in sizes; no live-state re-read.
        ctx.save_for_backward(x, source, amplitude_max, packed)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels

        x, source, amplitude_max, packed = ctx.saved_tensors
        b, ni, no, li, lo, oi, oo = ctx.sizes
        recipe, a = ctx.recipe, len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_zeros(x.shape) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        tiles = tr.cdiv(b, recipe.batch_tile)
        partial = x.new_empty((tiles, 3, a)) if need_p else None
        if a and (need_x or need_p):
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
                kernels.reduce_parameters[(tr.cdiv(a, recipe.prep_group),)](
                    partial,
                    dp,
                    source,
                    amplitude_max,
                    a,
                    tiles,
                    tr.next_power_of_2(tiles),
                    recipe.prep_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None


def onchip_h(x, p, value, sizes, recipe):
    return _OnchipH.apply(x, p, value, sizes, recipe)
