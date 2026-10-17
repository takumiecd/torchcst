"""First-order CUDA autograd with live preparation and shared moment banks."""

import torch
from torch.autograd.function import once_differentiable


def _basis(n, device):
    z = torch.arange(n, dtype=torch.float64, device=device) * (2.0 / (n - 1)) - 1
    return torch.stack((torch.ones_like(z), z, z.square()))


def _norm_moments(n, device):
    count = torch.arange(n + 1, device=device, dtype=torch.float64)
    safe = count.clamp_min(1)
    m2 = torch.where(count > 1, count / 3 - 1 / (3 * safe), 0)
    m4 = torch.where(count > 1, count / 5 - 2 / (3 * safe) + 7 / (15 * safe.pow(3)), 0)
    return torch.stack((m2, m4))


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
    def forward(ctx, x, p, value, chart, recipe, bases, moments):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from . import kernels as k

        no, ni = chart.grid_shape[0][0], chart.grid_shape[1][0]
        source = p.contiguous().clone()
        origin, spacing = chart.origin.clone(), chart.spacing.clone()
        meta = torch.empty((len(p), 9), device=p.device, dtype=torch.float64)
        ctx.matrix = False
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
                moments,
                moments.shape[1],
                recipe.reciprocal,
                recipe.norm_lookup,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if recipe.strategy == "matrix":
                ctx.matrix = True
                w = x.new_zeros((no, ni))
                k.matrix_build[(tr.cdiv(len(p), 8),)](
                    meta,
                    w,
                    origin,
                    spacing,
                    len(p),
                    no,
                    ni,
                    16,
                    8,
                    recipe.reciprocal,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                ctx.sizes = (no, ni)
                ctx.recipe = recipe
                ctx.save_for_backward(meta, x, w, origin, spacing)
                return x @ w.T
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

        if ctx.matrix:
            meta, x, w, origin, spacing = ctx.saved_tensors
            dy = dy.contiguous()
            dx = dy @ w if ctx.needs_input_grad[0] else None
            dp = None
            if ctx.needs_input_grad[1]:
                dw = dy.T @ x
                dp = dy.new_empty((len(meta), 4))
                k.matrix_vjp[(tr.cdiv(len(meta), 8),)](
                    meta,
                    dw,
                    dp,
                    origin,
                    spacing,
                    len(meta),
                    *ctx.sizes,
                    16,
                    8,
                    ctx.recipe.reciprocal,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return dx, dp, None, None, None, None, None
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
        return dx, dp, None, None, None, None, None


def apply(x, p, value, chart, recipe, bases, moments):
    return _BSpline.apply(x.contiguous(), p, value, chart, recipe, bases, moments)
