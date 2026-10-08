"""Retain bounded support lists and H; overflow profiles use all actual sites."""

import torch
from torch.autograd.function import once_differentiable

from torchcst._backends.torch.parameterizations import polar_amp_width as polar

from .executor import _geometry


def prepare(x, p, kernel, charts, recipe):
    """Snapshot geometry/Polar and pack complete supports for either contraction."""
    import triton

    from .support_kernels import pack

    x = x.contiguous()
    source = p.contiguous().clone()
    amp, alpha = polar._amplitude_and_alpha(kernel, source[:, :2])
    sigmas = polar._bandwidth_sigmas(kernel, amp, alpha)
    floors = tuple(b.normalization.floor for b in kernel.spec.profiles)
    cap = recipe.support_capacity
    sides = []
    for c, center, sigma, floor in zip(
        charts, (source[:, 2:4], source[:, 4:6]), sigmas, floors
    ):
        sites, q, jac = _geometry(c, center)
        precision = sigma.reciprocal().square().contiguous()
        index = torch.empty((len(p), cap), device=x.device, dtype=torch.int32)
        phi = x.new_empty((len(p), cap))
        norm = x.new_empty(len(p))
        count = torch.empty(len(p), device=x.device, dtype=torch.int32)
        if len(p):
            pack[(len(p),)](
                sites,
                q,
                precision,
                index,
                phi,
                norm,
                count,
                len(sites),
                cap,
                triton.next_power_of_2(len(sites)),
                floor,
                num_warps=4,
                enable_fp_fusion=False,
            )
        sides.append((sites, q, jac, precision, index, phi, norm, count))
    z = source[:, :2]
    square = z.square().sum(-1)
    safe = square.clamp_min(torch.finfo(source.dtype).tiny)
    damp = -amp[:, None] * z / safe[:, None]
    damp = torch.where((square >= torch.finfo(source.dtype).tiny)[:, None], damp, 0)
    damp[:, 0] += kernel.scalar("amplitude_max").to(source) / safe.sqrt()
    return x, source, amp, damp, floors, sides


class _Support(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, kernel, charts, recipe):
        import triton

        from .support_kernels import forward

        x, source, amp, damp, floors, sides = prepare(x, p, kernel, charts, recipe)
        cap = recipe.support_capacity
        need_p = ctx.needs_input_grad[1]
        h = x.new_empty((len(p), len(x))) if need_p else x.new_empty(0)
        y = x.new_zeros((len(x), len(sides[1][0])))
        if len(p):
            forward[(len(p),)](
                x,
                *[sides[0][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                *[sides[1][i] for i in (0, 1, 3, 4, 5, 6, 7)],
                amp,
                h,
                y,
                len(x),
                x.shape[1],
                y.shape[1],
                cap,
                triton.next_power_of_2(len(x)),
                64,
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

        from .support_kernels import backward

        x, p, amp, damp, h, *sides = ctx.saved_tensors
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = torch.zeros_like(x) if need_x else None
        dp = torch.empty_like(p) if need_p else None
        if len(p):
            backward[(len(p),)](
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
                64,
                *ctx.floors,
                need_x,
                need_p,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return dx, dp, None, None, None


def support_linear(x, p, kernel, charts, recipe):
    return _Support.apply(x, p, kernel, charts, recipe)
