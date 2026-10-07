"""Assemble current FP32 W, save forward snapshots, and form canonical VJPs."""

import torch
from torch.autograd.function import once_differentiable


def _matmul(left, right, recipe, *, allow_split=True):
    if getattr(recipe, "gemm", "torch") == "torch":
        return left @ right
    import triton as tr

    from .kernels import ieee_matmul

    m, k = left.shape
    n = right.shape[1]
    result = left.new_empty((m, n))
    requested = getattr(recipe, "split_k", 1) if allow_split else 1
    if requested > 1:
        from .recipe_v4 import split_count
        from .split_kernels import ieee_split_matmul, reduce_partials

        splits = split_count(k, requested)
        if splits > 1:
            partial = left.new_empty((splits, m, n))
            ieee_split_matmul[(tr.cdiv(m, 16), tr.cdiv(n, 32), splits)](
                left,
                right,
                partial,
                m,
                n,
                k,
                *left.stride(),
                *right.stride(),
                SPLITS=splits,
                BM=16,
                BN=32,
                BK=32,
                num_warps=4,
                enable_fp_fusion=False,
            )
            reduce_partials[(tr.cdiv(m * n, 256),)](
                partial,
                result,
                m * n,
                splits,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return result
    ieee_matmul[(tr.cdiv(m, 16), tr.cdiv(n, 32))](
        left,
        right,
        result,
        m,
        n,
        k,
        *left.stride(),
        *right.stride(),
        BM=16,
        BN=32,
        BK=32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


class _ProfileMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, pitch, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from ..profile_product_global.grouped_kernels import prepare
        from .kernels import assemble

        group = getattr(recipe, "atom_group", 1)
        patch_kwargs = {}
        if hasattr(recipe, "atom_group"):
            from .grouped_kernels import assemble

            patch_kwargs["GROUP"] = group

        b, ni, no, tile, spacing, oi, oo = sizes
        source = p.contiguous().clone()
        saved_pitch = pitch.clone() if pitch is not None else None
        scalars = polar_scalars(value)
        amplitude_max = scalars[0].clone()
        a = len(source)
        packed = p.new_empty((13, a))
        order = None
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
            if hasattr(recipe, "spatial_tile"):
                from .spatial_kernels import order_support

                packed, order = order_support(packed, ni, no, recipe.spatial_tile)
            assemble[(tr.cdiv(a, group),)](
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
                **patch_kwargs,
            )
            y = _matmul(x, w.T, recipe)
        else:
            y = x.new_zeros((b, no))
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(x, source, saved_pitch, amplitude_max, packed, w, order)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from .kernels import parameter_vjp

        group = getattr(ctx.recipe, "atom_group", 1)
        patch_kwargs = {}
        if hasattr(ctx.recipe, "atom_group"):
            from .grouped_kernels import parameter_vjp

            patch_kwargs["GROUP"] = group

        x, source, pitch, amplitude_max, packed, w, order = ctx.saved_tensors
        if order is not None:
            patch_kwargs.update(Order=order, ORDERED=True)
        _b, ni, no, tile, spacing, oi, oo = ctx.sizes
        a = len(source)
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = dp = None
        if not a:
            dx = torch.zeros_like(x) if need_x else None
            dp = torch.zeros_like(source) if need_p else None
        else:
            if need_x:
                dx = _matmul(dy, w, ctx.recipe)
            if need_p:
                # Batch K<=64 already has many output tiles. Avoid S*NO*NI scratch.
                dw = _matmul(dy.T, x, ctx.recipe, allow_split=False)
                dp = torch.empty_like(source)
                parameter_vjp[(tr.cdiv(a, group),)](
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
                    **patch_kwargs,
                )
        return dx, dp, None, None, None, None


def matrix_product(x, p, value, pitch, sizes, recipe):
    return _ProfileMatrix.apply(x, p, pitch, value, sizes, recipe)
