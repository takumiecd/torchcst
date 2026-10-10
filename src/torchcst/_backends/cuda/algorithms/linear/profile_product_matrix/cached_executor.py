"""Reuse exact norm/interval snapshots while bounding the expanded support view."""

import torch
from torch.autograd.function import once_differentiable

from .bounded_executor import _prepare
from .executor import _matmul


class _CachedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from .cache_kernels import encode
        from .grouped_kernels import assemble

        _, ni, no, _, spacing, oi, oo = sizes
        source = p.contiguous().clone()
        scalars = polar_scalars(value)
        amplitude_max = scalars[0].clone()
        a = len(source)
        factors = p.new_empty((5, a))
        ends = torch.empty((4, a), device=p.device, dtype=torch.int16)
        flags = torch.empty(a, device=p.device, dtype=torch.uint8)
        packed = p.new_empty((13, min(a, recipe.atom_chunk)))
        w = p.new_zeros((no, ni))
        for start in range(0, a, recipe.atom_chunk):
            chunk = source[start : start + recipe.atom_chunk]
            ca = len(chunk)
            _prepare(
                chunk, packed, scalars, value.spec.normalization.floor, sizes, recipe
            )
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
            encode[(tr.cdiv(ca, 256),)](
                packed,
                factors,
                ends,
                flags,
                ca,
                a,
                start,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        y = _matmul(x, w.T, recipe) if a else x.new_zeros((len(x), no))
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(x, source, w, amplitude_max, factors, ends, flags)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from .cache_kernels import decode
        from .grouped_kernels import parameter_vjp

        x, source, w, amplitude_max, factors, ends, flags = ctx.saved_tensors
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
                decode[(tr.cdiv(ca, 256),)](
                    factors,
                    ends,
                    flags,
                    chunk,
                    amplitude_max,
                    packed,
                    ca,
                    a,
                    start,
                    256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                parameter_vjp[(tr.cdiv(ca, recipe.atom_group),)](
                    packed,
                    dw,
                    dp[start : start + ca],
                    chunk,
                    amplitude_max,
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


def cached_product(x, p, value, sizes, recipe):
    return _CachedProduct.apply(x, p, value, sizes, recipe)
