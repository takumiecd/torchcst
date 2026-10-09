"""Original support preparation with lossless int16 storage and shared W/dW."""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.algorithms.linear.torus_profile_product.executor import (
    _queries,
    _Scalars,
)
from torchcst._backends.torch.parameterizations import polar_amp_width as polar


class _Weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, chart, recipe):
        import triton
        from ..sparse_weight.kernels import pack
        from .contractions import contract

        x, source = x.contiguous(), p.contiguous().clone()
        scalars = _Scalars(kernel, source)
        amp, alpha = polar._amplitude_and_alpha(scalars, source[:, :2])
        sigma, _, _ = polar._sigma_bounds(scalars, amp, alpha, side="input")
        precision = sigma.reciprocal().square().detach()
        major, minor, circle, sites, axis = _queries(chart, source)
        assert axis == 0
        maximum = scalars.scalar("amplitude_max")
        spacing = chart.axes[0].spacing.to(source).clone()
        start = chart.axes[0].start.to(source).clone()
        pitch = chart.tile_pitch.to(source).clone()
        ni, no, atoms = x.shape[1], chart.shape[0], len(p)
        sides = []
        for side, n, cap in (
            (0, no, recipe.circle_capacity),
            (1, ni, recipe.section_capacity),
        ):
            index = torch.empty((atoms, cap), device=x.device, dtype=torch.int16)
            raw, stats = x.new_empty((atoms, cap)), x.new_empty((atoms, 5))
            if atoms:
                pack[(atoms,)](
                    source,
                    precision,
                    maximum,
                    major,
                    minor,
                    circle,
                    sites,
                    index,
                    raw,
                    stats,
                    start,
                    spacing,
                    pitch,
                    N=n,
                    CAP=cap,
                    V=triton.next_power_of_2(n),
                    TILE=chart.tile_shape[0],
                    SIDE=side,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            sides.extend((index, raw, stats))
        w = x.new_zeros((no, ni))
        if atoms:
            contract[(atoms,)](
                w,
                source,
                precision,
                maximum,
                major,
                minor,
                circle,
                sites,
                *sides,
                source,
                NI=ni,
                NO=no,
                CO=recipe.circle_capacity,
                CI=recipe.section_capacity,
                T=recipe.patch_tile,
                FLOOR=kernel.spec.normalization.floor,
                VJP=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.recipe, ctx.floor = recipe, kernel.spec.normalization.floor
        ctx.save_for_backward(
            x, source, precision, maximum, major, minor, circle, sites, w, *sides
        )
        return x @ w.T

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from .contractions import contract

        x, source, precision, maximum, major, minor, circle, sites, w, *sides = (
            ctx.saved_tensors
        )
        need_x, need_p = ctx.needs_input_grad[:2]
        dy = dy.contiguous()
        dx = dy @ w if need_x else None
        dp = torch.empty_like(source) if need_p else None
        if need_p and len(source):
            dw = dy.T @ x
            recipe = ctx.recipe
            contract[(len(source),)](
                dw,
                source,
                precision,
                maximum,
                major,
                minor,
                circle,
                sites,
                *sides,
                dp,
                NI=x.shape[1],
                NO=dy.shape[1],
                CO=recipe.circle_capacity,
                CI=recipe.section_capacity,
                T=recipe.patch_tile,
                FLOOR=ctx.floor,
                VJP=True,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def compact_ids(x, p, kernel, chart, recipe):
    return _Weight.apply(x, p, kernel, chart, recipe)
