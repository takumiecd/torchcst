"""Retain W and packed supports, then reuse a shared dense dW in all atom VJPs."""

import torch
from torch.autograd.function import once_differentiable

from .fused_prepare import prepare


class _GroupedWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe):
        import triton

        from .grouped_kernels import assemble

        x, source, amp, damp, floors, sides = prepare(
            x,
            p,
            kernel,
            charts,
            recipe,
            index_dtype=torch.int16 if recipe.index_bits == 16 else torch.int32,
        )
        w = x.new_zeros((len(sides[1][0]), x.shape[1]))
        if len(p):
            assemble[(triton.cdiv(len(p), recipe.atom_group),)](
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                w,
                x.shape[1],
                w.shape[0],
                recipe.support_capacity,
                recipe.patch_tile,
                len(p),
                recipe.atom_group,
                *floors,
                num_warps=4,
                enable_fp_fusion=False,
            )
        y = x @ w.T
        ctx.recipe, ctx.floors = recipe, floors
        ctx.save_for_backward(x, source, amp, damp, w, *sides[0], *sides[1])
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from .grouped_kernels import vjp

        x, p, amp, damp, w, *sides = ctx.saved_tensors
        dy = dy.contiguous()
        dx = dy @ w if ctx.needs_input_grad[0] else None
        dp = None
        if ctx.needs_input_grad[1]:
            dw = (dy.T @ x).contiguous()
            dp = torch.empty_like(p)
            if len(p):
                vjp[(triton.cdiv(len(p), ctx.recipe.atom_group),)](
                    *sides,
                    amp,
                    damp,
                    dw,
                    dp,
                    x.shape[1],
                    dy.shape[1],
                    ctx.recipe.support_capacity,
                    ctx.recipe.patch_tile,
                    len(p),
                    ctx.recipe.atom_group,
                    *ctx.floors,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None


def grouped_weight_linear(x, p, kernel, charts, recipe):
    return _GroupedWeight.apply(x, p, kernel, charts, recipe)
