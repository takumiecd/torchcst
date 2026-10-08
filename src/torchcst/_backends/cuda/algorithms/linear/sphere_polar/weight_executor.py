"""Retain W and packed supports, then reuse a shared dense dW in all atom VJPs."""

import torch
from torch.autograd.function import once_differentiable

from .support_executor import prepare


class _Weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe):
        from .weight_kernels import assemble

        x, source, amp, damp, floors, sides = prepare(x, p, kernel, charts, recipe)
        w = x.new_zeros((len(sides[1][0]), x.shape[1]))
        if len(p):
            assemble[(len(p),)](
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                w,
                x.shape[1],
                w.shape[0],
                recipe.support_capacity,
                64,
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
        from .weight_kernels import vjp

        x, p, amp, damp, w, *sides = ctx.saved_tensors
        dy = dy.contiguous()
        dx = dy @ w if ctx.needs_input_grad[0] else None
        dp = None
        if ctx.needs_input_grad[1]:
            dw = (dy.T @ x).contiguous()
            dp = torch.empty_like(p)
            if len(p):
                vjp[(len(p),)](
                    *sides,
                    amp,
                    damp,
                    dw,
                    dp,
                    x.shape[1],
                    dy.shape[1],
                    ctx.recipe.support_capacity,
                    64,
                    *ctx.floors,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None


def weight_linear(x, p, kernel, charts, recipe):
    return _Weight.apply(x, p, kernel, charts, recipe)
