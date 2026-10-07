"""Assemble current FP32 W, save forward snapshots, and form canonical VJPs."""

import torch
from torch.autograd.function import once_differentiable


class _ProfileMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, pitch, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from ..profile_product_global.grouped_kernels import prepare
        from .kernels import assemble

        b, ni, no, tile, spacing, oi, oo = sizes
        source = p.contiguous().clone()
        saved_pitch = pitch.clone() if pitch is not None else None
        scalars = polar_scalars(value)
        amplitude_max = scalars[0].clone()
        a = len(source)
        packed = p.new_empty((13, a))
        w = p.new_zeros((no, ni))
        if a:
            prepare[(tr.cdiv(a, recipe.prep_group),)](
                source,
                packed,
                a,
                ni,
                no,
                spacing,
                oi,
                oo,
                tr.next_power_of_2(ni),
                tr.next_power_of_2(no),
                scalars,
                value.spec.normalization.floor,
                BOUNDS=True,
                STRIP_TILE=tile,
                Pitch=saved_pitch,
                GROUP=recipe.prep_group,
                SUPPORT=recipe.preparation == "support",
                PREP_SITES=recipe.prep_sites,
                num_warps=4,
                enable_fp_fusion=False,
            )
            assemble[(a,)](
                packed,
                w,
                saved_pitch,
                a,
                ni,
                no,
                tile,
                spacing,
                oi,
                oo,
                recipe.patch_sites,
                num_warps=4,
                enable_fp_fusion=False,
            )
            y = x @ w.T
        else:
            y = x.new_zeros((b, no))
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(x, source, saved_pitch, amplitude_max, packed, w)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from .kernels import parameter_vjp

        x, source, pitch, amplitude_max, packed, w = ctx.saved_tensors
        _b, ni, no, tile, spacing, oi, oo = ctx.sizes
        a = len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = dp = None
        if not a:
            dx = torch.zeros_like(x) if need_x else None
            dp = torch.zeros_like(source) if need_p else None
        else:
            if need_x:
                dx = dy @ w
            if need_p:
                dw = dy.T @ x
                dp = torch.empty_like(source)
                parameter_vjp[(a,)](
                    packed,
                    dw,
                    dp,
                    source,
                    amplitude_max,
                    pitch,
                    a,
                    ni,
                    no,
                    tile,
                    spacing,
                    oi,
                    oo,
                    *dw.stride(),
                    ctx.recipe.patch_sites,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None, None


def matrix_product(x, p, value, pitch, sizes, recipe):
    return _ProfileMatrix.apply(x, p, pitch, value, sizes, recipe)
