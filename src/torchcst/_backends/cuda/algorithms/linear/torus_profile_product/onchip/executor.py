"""Snapshot operands and optionally retain compact H/input-VJP data for backward."""

from dataclasses import replace

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
    _queries,
    _Scalars,
)
from torchcst._backends.torch.parameterizations import polar_amp_width as polar


class _Onchip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, chart, recipe):
        import triton

        from .kernels import forward

        x = x.contiguous()
        source = p.contiguous().clone()
        scalars = _Scalars(kernel, source)
        amplitude, alpha = polar._amplitude_and_alpha(scalars, source[:, :2])
        sigma, _, _ = polar._sigma_bounds(scalars, amplitude, alpha, side="input")
        precision = sigma.reciprocal().square().detach()
        major, minor, circle, sites, axis = _queries(chart, source)
        assert axis == 0
        maximum = scalars.scalar("amplitude_max")
        y = x.new_zeros((len(x), chart.shape[0]))
        save_mode = recipe.save
        if save_mode != "recompute" and not ctx.needs_input_grad[1]:
            save_mode = "norm" if ctx.needs_input_grad[0] else "recompute"
        k, c = (4, 11) if save_mode == "vjp" else (1, 6)
        h_cache = (
            x.new_empty((len(p), k, len(x)))
            if save_mode in ("h", "vjp")
            else x.new_empty(0)
        )
        info = x.new_empty((len(p), c)) if save_mode != "recompute" else x.new_empty(0)
        if len(p):
            forward[(len(p),)](
                x,
                source,
                precision,
                maximum,
                major,
                minor,
                circle,
                sites,
                h_cache,
                info,
                y,
                B=len(x),
                NI=x.shape[1],
                NO=y.shape[1],
                PB=triton.next_power_of_2(len(x)),
                T=recipe.site_tile,
                TRIG=recipe.trig,
                SAVE=save_mode,
                FLOOR=kernel.spec.normalization.floor,
                num_warps=recipe.num_warps,
                enable_fp_fusion=False,
            )
        ctx.recipe = recipe
        ctx.save_mode = save_mode
        ctx.input_shape = tuple(x.shape)
        ctx.floor = kernel.spec.normalization.floor
        # All dependence on X is already represented by saved H/input-VJP
        # contractions; backward-vjp never loads the X pointer.
        saved_x = x.new_empty(0) if save_mode == "vjp" else x
        ctx.save_for_backward(
            saved_x,
            source,
            precision,
            maximum,
            major,
            minor,
            circle,
            sites,
            h_cache,
            info,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from .kernels import backward

        x, p, precision, maximum, major, minor, circle, sites, h_cache, info = (
            ctx.saved_tensors
        )
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_zeros(ctx.input_shape) if need_x else None
        dp = torch.empty_like(p) if need_p else None
        if len(p) and (need_x or need_p):
            backward[(len(p),)](
                x,
                dy.contiguous(),
                p,
                precision,
                maximum,
                major,
                minor,
                circle,
                sites,
                h_cache,
                info,
                dx if need_x else x,
                dp if need_p else p,
                B=ctx.input_shape[0],
                NI=ctx.input_shape[1],
                NO=dy.shape[1],
                PB=triton.next_power_of_2(ctx.input_shape[0]),
                T=ctx.recipe.site_tile,
                TRIG=ctx.recipe.trig,
                SAVE=ctx.save_mode,
                FLOOR=ctx.floor,
                NEED_X=need_x,
                NEED_P=need_p,
                num_warps=ctx.recipe.num_warps,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def onchip_product(x, p, kernel, chart, recipe):
    if not torch.is_grad_enabled() and recipe.save != "recompute":
        recipe = replace(recipe, save="recompute")
    return _Onchip.apply(x, p, kernel, chart, recipe)
