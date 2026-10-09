"""Save compact profile snapshots and H; keep G inside direct CUDA Core CTAs."""

import torch
from torch.autograd.function import once_differentiable

from .fused_prepare import prepare


class _Direct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe):
        import triton

        from .direct_kernels import forward

        x, source, amp, damp, floors, sides = prepare(
            x, p, kernel, charts, recipe, index_dtype=torch.int16
        )
        need_p = ctx.needs_input_grad[1]
        h = x.new_empty((len(p), len(x))) if need_p else x.new_empty(0)
        y = x.new_zeros((len(x), len(sides[1][0])))
        if len(p):
            forward[(triton.cdiv(len(p), recipe.atom_group),)](
                x,
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                h,
                y,
                len(x),
                x.shape[1],
                y.shape[1],
                recipe.support_capacity,
                triton.next_power_of_2(len(x)),
                recipe.support_tile,
                len(p),
                recipe.atom_group,
                *floors,
                need_p,
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.recipe, ctx.floors = recipe, floors
        ctx.save_for_backward(x, source, amp, damp, h, *sides[0], *sides[1])
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton

        from .direct_kernels import backward

        x, p, amp, damp, h, *sides = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_x else None
        dp = torch.empty_like(p) if need_p else None
        if len(p):
            backward[(triton.cdiv(len(p), ctx.recipe.atom_group),)](
                x,
                dy.contiguous(),
                *sides,
                amp,
                damp,
                h,
                dx if need_x else x,
                dp if need_p else p,
                len(x),
                x.shape[1],
                dy.shape[1],
                ctx.recipe.support_capacity,
                triton.next_power_of_2(len(x)),
                ctx.recipe.support_tile,
                len(p),
                ctx.recipe.atom_group,
                *ctx.floors,
                need_x,
                need_p,
                ctx.recipe.merge_output_vjp,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def direct_linear(x, p, kernel, charts, recipe):
    return _Direct.apply(x, p, kernel, charts, recipe)
