"""Identical preparation, snapshots and first-order VJPs for two contractions."""

import torch
from torch.autograd.function import once_differentiable

from ..profile_product_matrix.executor import _matmul


def _prepare(source, value, sizes, recipe):
    import triton as tr

    from ..local_product.preparation import polar_scalars
    from . import kernels as k

    _b, ni, no, li, lo, oi, oo = sizes
    a = len(source)
    packed = source.new_empty((13, a))
    if a:
        k.prepare[(tr.cdiv(a, recipe.prep_group),)](
            source,
            packed,
            a,
            ni,
            no,
            li,
            lo,
            oi,
            oo,
            polar_scalars(value),
            value.spec.normalization.floor,
            recipe.prep_group,
            recipe.prep_sites,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return packed


def _axis(x, packed, sizes, recipe, *, output):
    import triton as tr

    from . import kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    result = x.new_empty((a, b))
    if a:
        k.axis_contract[(tr.cdiv(a, recipe.atom_group),)](
            x,
            packed,
            result,
            a,
            b,
            no if output else ni,
            lo if output else li,
            oo if output else oi,
            *x.stride(),
            output,
            max(16, tr.next_power_of_2(b)),
            recipe.patch_sites,
            recipe.atom_group,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result


def _scatter(t, packed, sizes, recipe, *, output):
    import triton as tr

    from . import kernels as k

    b, ni, no, li, lo, oi, oo = sizes
    a = packed.shape[1]
    result = t.new_zeros((b, no if output else ni))
    if a:
        k.axis_scatter[(tr.cdiv(a, recipe.atom_group),)](
            t,
            packed,
            result,
            a,
            b,
            no if output else ni,
            lo if output else li,
            oo if output else oi,
            output,
            max(16, tr.next_power_of_2(b)),
            recipe.patch_sites,
            recipe.atom_group,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result


class _Periodic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe, mode):
        import triton as tr

        from . import kernels as k

        _b, ni, no, li, lo, oi, oo = sizes
        source = p.contiguous().clone()
        amplitude_max = value.scalar("amplitude_max").clone()
        packed = _prepare(source, value, sizes, recipe)
        a = len(source)
        w = h = None
        if mode == "matrix":
            w = p.new_zeros((no, ni))
            if a:
                k.assemble[(tr.cdiv(a, recipe.atom_group),)](
                    packed,
                    w,
                    a,
                    ni,
                    no,
                    li,
                    lo,
                    oi,
                    oo,
                    recipe.patch_sites,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            y = _matmul(x, w.T, recipe)
        else:
            h = _axis(x, packed, sizes, recipe, output=False)
            y = _scatter(h, packed, sizes, recipe, output=True)
        ctx.sizes, ctx.recipe, ctx.mode = sizes, recipe, mode
        ctx.save_for_backward(x, source, amplitude_max, packed, w, h)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        x, source, amplitude_max, packed, w, h = ctx.saved_tensors
        b, ni, no, li, lo, oi, oo = ctx.sizes
        a = len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        recipe = ctx.recipe
        dx = dp = None
        if not a:
            dx = torch.zeros_like(x) if need_x else None
            dp = torch.zeros_like(source) if need_p else None
        elif ctx.mode == "matrix":
            if need_x:
                dx = _matmul(dy, w, recipe)
            if need_p:
                dw = _matmul(dy.T, x, recipe, allow_split=False)
                dp = torch.empty_like(source)
                k.parameter_vjp[(tr.cdiv(a, recipe.atom_group),)](
                    packed,
                    dw,
                    dp,
                    source,
                    amplitude_max,
                    a,
                    ni,
                    no,
                    li,
                    lo,
                    oi,
                    oo,
                    *dw.stride(),
                    recipe.patch_sites,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        elif need_x or need_p:
            g = _axis(dy, packed, ctx.sizes, recipe, output=True)
            if need_x:
                dx = _scatter(g, packed, ctx.sizes, recipe, output=False)
            if need_p:
                dp = torch.empty_like(source)
                k.factor_vjp[(tr.cdiv(a, recipe.atom_group),)](
                    x,
                    dy,
                    packed,
                    h,
                    g,
                    dp,
                    source,
                    amplitude_max,
                    a,
                    b,
                    ni,
                    no,
                    li,
                    lo,
                    oi,
                    oo,
                    *x.stride(),
                    *dy.stride(),
                    max(16, tr.next_power_of_2(b)),
                    recipe.patch_sites,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None, None


def periodic_product(x, p, value, sizes, recipe, *, mode):
    if mode not in ("matrix", "factor"):
        raise ValueError("unsupported periodic contraction")
    return _Periodic.apply(x, p, value, sizes, recipe, mode)
