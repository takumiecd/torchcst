"""Matched baseline snapshots/forward; only the H/dH producer shares X."""

from dataclasses import asdict

import torch
from torch.autograd.function import once_differentiable

from .executor import _OnchipH
from .recipe import InputBatchProducerHRecipe


def baseline_recipe(recipe):
    values = asdict(recipe)
    values.pop("panel_sites")
    return InputBatchProducerHRecipe(**values)


class _SharedXH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        ctx.panel_sites = recipe.panel_sites
        return _OnchipH.forward(ctx, x, p, value, sizes, baseline_recipe(recipe))

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        need_x, need_p = ctx.needs_input_grad[:2]
        if not (need_x and need_p):
            return _OnchipH.backward(ctx, dy)
        import triton as tr

        from . import kernels, shared_x_kernels
        from .input_owner import input_owned_backward

        x, source, amplitude_max, packed = ctx.saved_tensors
        b = ctx.sizes[0]
        recipe, a = ctx.recipe, len(source)
        dx = x.new_zeros(x.shape)
        dp = torch.empty_like(source)
        partial_tiles = tr.cdiv(min(b, recipe.g_batch), recipe.producer_batch_tile)
        partial = x.new_empty((partial_tiles, 3, a))
        physical = x.new_empty((3, a))
        if a:
            input_owned_backward(
                x,
                dy,
                packed,
                ctx.sizes,
                recipe,
                dx,
                partial,
                True,
                physical=physical,
                producer=shared_x_kernels.produce_g_parameters_shared_x,
                producer_options={"PANEL": ctx.panel_sites},
            )
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


def shared_x_h(x, p, value, sizes, recipe):
    return _SharedXH.apply(x, p, value, sizes, recipe)
