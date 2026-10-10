"""Reuse support scratch across chunks; recompute from immutable forward state."""

import torch
from torch.autograd.function import once_differentiable

from .executor import _matmul


def _prepare(source, packed, scalars, floor, sizes, recipe):
    import triton as tr

    from ..profile_product_global.grouped_kernels import prepare

    _, ni, no, _, spacing, oi, oo = sizes
    a = len(source)
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
        floor,
        BOUNDS=True,
        STRIP_TILE=0,
        Pitch=None,
        GROUP=recipe.prep_group,
        SUPPORT=recipe.preparation == "support",
        PREP_SITES=recipe.prep_sites,
        num_warps=4,
        enable_fp_fusion=False,
    )


class _BoundedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from .grouped_kernels import assemble

        _, ni, no, _, spacing, oi, oo = sizes
        source = p.contiguous().clone()
        # All decoder values are needed again in backward. Later live scalar
        # changes and another forward must not change this call's support/norm.
        scalars = tuple(s.clone() for s in polar_scalars(value))
        floor = value.spec.normalization.floor
        a = len(source)
        packed = p.new_empty((13, min(a, recipe.atom_chunk)))
        w = p.new_zeros((no, ni))
        for start in range(0, a, recipe.atom_chunk):
            chunk = source[start : start + recipe.atom_chunk]
            ca = len(chunk)
            _prepare(chunk, packed, scalars, floor, sizes, recipe)
            assemble[(tr.cdiv(ca, recipe.atom_group),)](
                packed,
                w,
                None,
                ca,
                ni,
                no,
                0,
                spacing,
                oi,
                oo,
                recipe.patch_sites,
                GROUP=recipe.atom_group,
                num_warps=4,
                enable_fp_fusion=False,
            )
        y = _matmul(x, w.T, recipe) if a else x.new_zeros((len(x), no))
        ctx.sizes, ctx.recipe, ctx.floor = sizes, recipe, floor
        # The mutable packed scratch is neither retained nor shared with VJP.
        ctx.save_for_backward(x, source, w, *scalars)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from .grouped_kernels import parameter_vjp

        x, source, w, *scalars = ctx.saved_tensors
        _, ni, no, _, spacing, oi, oo = ctx.sizes
        recipe, a = ctx.recipe, len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = dp = None
        if not a:
            return (
                torch.zeros_like(x) if need_x else None,
                torch.zeros_like(source) if need_p else None,
                None,
                None,
                None,
            )
        if need_x:
            dx = _matmul(dy, w, recipe)
        if need_p:
            dw = _matmul(dy.T, x, recipe, allow_split=False)
            dp = torch.empty_like(source)
            packed = source.new_empty((13, min(a, recipe.atom_chunk)))
            for start in range(0, a, recipe.atom_chunk):
                chunk = source[start : start + recipe.atom_chunk]
                ca = len(chunk)
                _prepare(chunk, packed, tuple(scalars), ctx.floor, ctx.sizes, recipe)
                parameter_vjp[(tr.cdiv(ca, recipe.atom_group),)](
                    packed,
                    dw,
                    dp[start : start + ca],
                    chunk,
                    scalars[0],
                    None,
                    ca,
                    ni,
                    no,
                    0,
                    spacing,
                    oi,
                    oo,
                    *dw.stride(),
                    recipe.patch_sites,
                    GROUP=recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return dx, dp, None, None, None


def bounded_product(x, p, value, sizes, recipe):
    return _BoundedProduct.apply(x, p, value, sizes, recipe)
