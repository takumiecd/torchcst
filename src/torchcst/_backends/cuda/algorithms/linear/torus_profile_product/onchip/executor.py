"""Snapshot operands; H/G and their derivative contractions remain inside a CTA."""

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
                y,
                B=len(x),
                NI=x.shape[1],
                NO=y.shape[1],
                PB=triton.next_power_of_2(len(x)),
                T=recipe.site_tile,
                FLOOR=kernel.spec.normalization.floor,
                num_warps=recipe.num_warps,
                enable_fp_fusion=False,
            )
        ctx.recipe = recipe
        ctx.floor = kernel.spec.normalization.floor
        ctx.save_for_backward(
            x, source, precision, maximum, major, minor, circle, sites
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from .kernels import backward

        x, p, precision, maximum, major, minor, circle, sites = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_x else None
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
                dx if need_x else x,
                dp if need_p else p,
                B=len(x),
                NI=x.shape[1],
                NO=dy.shape[1],
                PB=triton.next_power_of_2(len(x)),
                T=ctx.recipe.site_tile,
                FLOOR=ctx.floor,
                NEED_X=need_x,
                NEED_P=need_p,
                num_warps=ctx.recipe.num_warps,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def onchip_product(x, p, kernel, chart, recipe):
    return _Onchip.apply(x, p, kernel, chart, recipe)
