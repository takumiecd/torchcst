"""First-order CUDA autograd with live preparation and shared moment banks."""

import torch
from torch.autograd.function import once_differentiable


def _basis(n, device):
    z = torch.arange(n, dtype=torch.float64, device=device) * (2.0 / (n - 1)) - 1
    return torch.stack((torch.ones_like(z), z, z.square()))


def _bank(x, basis):
    bank = torch.empty(
        (3, len(x), x.shape[1] + 1), dtype=torch.float64, device=x.device
    )
    bank[:, :, 0] = 0
    torch.cumsum(x.double()[None] * basis[:, None], -1, out=bank[:, :, 1:])
    return bank


def _query(x, meta, origin, spacing, dim, recipe, basis):
    import triton as tr

    from . import kernels as k

    b, n = x.shape
    a = len(meta)
    bank = x if recipe.strategy == "direct" else _bank(x, basis)
    h = x.new_empty((b, a))
    hp = torch.empty_like(h)
    k.query[(a, tr.cdiv(b, 16))](
        x,
        bank,
        meta,
        h,
        hp,
        origin,
        spacing,
        a,
        b,
        n,
        dim,
        recipe.strategy == "direct",
        recipe.direct_sites,
        recipe.block_sites,
        recipe.reciprocal,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return h, hp


def _scatter(h, meta, origin, spacing, dim, n, recipe):
    import triton as tr

    from . import kernels as k

    b, a = h.shape
    y = h.new_zeros((b, n))
    delta = (
        y
        if recipe.strategy == "direct"
        else torch.zeros((3, b, n + 1), device=h.device, dtype=torch.float64)
    )
    k.inject[(a, tr.cdiv(b, 16))](
        h,
        meta,
        delta,
        y,
        origin,
        spacing,
        a,
        b,
        n,
        dim,
        recipe.strategy == "direct",
        recipe.direct_sites,
        recipe.block_sites,
        recipe.reciprocal,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if recipe.strategy != "direct":
        scan = delta.cumsum(-1)
        k.restore[(tr.cdiv(b * n, 128),)](
            scan, y, n, b, num_warps=4, enable_fp_fusion=False
        )
    return y


class _BSpline(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, bases):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from . import kernels as k

        no, ni = chart.grid_shape[0][0], chart.grid_shape[1][0]
        source = p.contiguous().clone()
        origin, spacing = chart.origin.clone(), chart.spacing.clone()
        meta = torch.empty((len(p), 9), device=p.device, dtype=torch.float64)
        if len(p):
            k.prepare[(tr.cdiv(len(p), 64),)](
                source,
                meta,
                origin,
                spacing,
                polar_scalars(value),
                len(p),
                no,
                ni,
                value.spec.normalization.floor,
                num_warps=4,
                enable_fp_fusion=False,
            )
            h, hp = _query(x, meta, origin, spacing, 1, recipe, bases[1])
            y = _scatter(h, meta, origin, spacing, 0, no, recipe)
        else:
            h = x.new_empty((len(x), 0))
            hp = torch.empty_like(h)
            y = x.new_zeros((len(x), no))
        ctx.recipe, ctx.bases, ctx.sizes = recipe, bases, (no, ni)
        ctx.save_for_backward(meta, h, hp, origin, spacing)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        meta, h, hp, origin, spacing = ctx.saved_tensors
        dy = dy.contiguous()
        a = len(meta)
        b = len(dy)
        dx = dp = None
        if a:
            g, gp = _query(dy, meta, origin, spacing, 0, ctx.recipe, ctx.bases[0])
            if ctx.needs_input_grad[0]:
                dx = _scatter(g, meta, origin, spacing, 1, ctx.sizes[1], ctx.recipe)
            if ctx.needs_input_grad[1]:
                dp = dy.new_empty((a, 4))
                k.vjp[(a,)](
                    h,
                    hp,
                    g,
                    gp,
                    meta,
                    dp,
                    a,
                    b,
                    tr.next_power_of_2(b),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            dx = dy.new_zeros((b, ctx.sizes[1])) if ctx.needs_input_grad[0] else None
            dp = dy.new_empty((0, 4)) if ctx.needs_input_grad[1] else None
        return dx, dp, None, None, None, None


def apply(x, p, value, chart, recipe, bases):
    return _BSpline.apply(x.contiguous(), p, value, chart, recipe, bases)
