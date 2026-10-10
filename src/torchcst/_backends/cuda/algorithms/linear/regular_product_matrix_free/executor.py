"""W-free exact Product contraction with chunk-bounded H/G and CSR scratch."""

import torch
from torch.autograd.function import once_differentiable


def _scratch(x, a, sizes, recipe):
    import triton as tr

    b, ni, no, *_ = sizes
    owners = max(tr.cdiv(ni, 16), tr.cdiv(no, 16))
    c = min(a, recipe.atom_chunk)

    def ints(n):
        return torch.empty(n, device=x.device, dtype=torch.int32)

    return (
        x.new_empty(c * b),
        x.new_empty(c * b),
        ints(owners + 1),
        ints(owners + 1),
        ints(owners),
        ints(recipe.owner_capacity * c),
        ints(c),
        x.new_empty(recipe.owner_splits * b * max(ni, no)),
    )


def _input_h(x, packed, h, sizes, recipe):
    import triton as tr

    from ..profile_product_global import grouped_kernels as grouped

    b, ni, _, _, spacing, oi, _ = sizes
    a = packed.numel() // 13
    grouped.input_h[(tr.cdiv(a, recipe.atom_group),)](
        x,
        packed,
        None,
        h,
        a,
        b,
        ni,
        0,
        spacing,
        oi,
        *x.stride(),
        max(16, tr.next_power_of_2(b)),
        8,
        recipe.atom_group,
        num_warps=4,
        enable_fp_fusion=False,
    )


def _contract(t, packed, destination, scratch, sizes, recipe, swap):
    import triton as tr

    from . import kernels as k

    _, _, counts, offsets, cursors, ids, overflow, partial = scratch
    b, ni, no, _, spacing, oi, oo = sizes
    n, origin = (ni, oi) if swap else (no, oo)
    owners, a = tr.cdiv(n, 16), packed.numel() // 13
    counts.zero_()
    k.count_members[(tr.cdiv(a, 256),)](
        packed,
        counts,
        overflow,
        a,
        owners,
        swap,
        recipe.owner_capacity,
        256,
        num_warps=4,
    )
    k.prefix[(1,)](
        counts, offsets, cursors, owners, tr.next_power_of_2(owners), num_warps=4
    )
    k.scatter_members[(tr.cdiv(a, 256),)](
        packed, cursors, ids, a, swap, recipe.owner_capacity, 256, num_warps=4
    )
    k.contract[(tr.cdiv(b, 16), owners, recipe.owner_splits)](
        t,
        packed,
        counts,
        offsets,
        ids,
        overflow,
        partial,
        a,
        b,
        n,
        spacing,
        origin,
        swap,
        recipe.owner_atoms,
        recipe.owner_splits,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.accumulate[(tr.cdiv(b * n, 256),)](
        partial,
        destination,
        b * n,
        recipe.owner_splits,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )


class _MatrixFree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from ..profile_product_global import grouped_kernels as grouped

        b, ni, no, _, spacing, oi, oo = sizes
        source = p.contiguous().clone()
        scalars = polar_scalars(value)
        amplitude_max = scalars[0].clone()
        a = len(source)
        # Each chunk is a contiguous 13-field SoA; no per-site matrices.
        saved = p.new_empty(13 * a)
        y = x.new_zeros((b, no))
        if a:
            scratch = _scratch(x, a, sizes, recipe)
            h = scratch[0]
            for start in range(0, a, recipe.atom_chunk):
                count = min(recipe.atom_chunk, a - start)
                packed = saved[13 * start : 13 * (start + count)]
                grouped.prepare[(tr.cdiv(count, recipe.prep_group),)](
                    source[start : start + count],
                    packed,
                    count,
                    ni,
                    no,
                    spacing,
                    oi,
                    oo,
                    tr.next_power_of_2(ni),
                    tr.next_power_of_2(no),
                    scalars,
                    value.spec.normalization.floor,
                    GROUP=recipe.prep_group,
                    SUPPORT=recipe.preparation == "support",
                    PREP_SITES=recipe.prep_sites,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                _input_h(x, packed, h, sizes, recipe)
                _contract(h, packed, y, scratch, sizes, recipe, False)
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(x, source, amplitude_max, saved)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels as k

        fused = getattr(ctx.recipe, "fused_backward", False)
        if fused:
            from . import fused_kernels

            backward_atoms = fused_kernels.backward_atoms
        else:
            backward_atoms = k.backward_atoms

        x, source, amplitude_max, saved = ctx.saved_tensors
        recipe, sizes = ctx.recipe, ctx.sizes
        b, ni, no, _, spacing, oi, oo = sizes
        need_x, need_p = ctx.needs_input_grad[:2]
        dx = x.new_zeros((b, ni)) if need_x else None
        dp = torch.empty_like(source) if need_p else None
        a = len(source)
        if a and (need_x or need_p):
            scratch = _scratch(x, a, sizes, recipe)
            h, g = scratch[:2]
            for start in range(0, a, recipe.atom_chunk):
                count = min(recipe.atom_chunk, a - start)
                packed = saved[13 * start : 13 * (start + count)]
                if need_p and not fused:
                    _input_h(x, packed, h, sizes, recipe)
                backward_atoms[(tr.cdiv(count, recipe.atom_group),)](
                    x,
                    dy,
                    packed,
                    h,
                    g,
                    dp[start : start + count] if need_p else None,
                    source[start : start + count],
                    amplitude_max,
                    None,
                    count,
                    b,
                    ni,
                    no,
                    0,
                    spacing,
                    oi,
                    oo,
                    *x.stride(),
                    *dy.stride(),
                    max(16, tr.next_power_of_2(b)),
                    8,
                    need_p,
                    recipe.atom_group,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                if need_x:
                    _contract(g, packed, dx, scratch, sizes, recipe, True)
        return dx, dp, None, None, None


def grid_product(x, p, value, sizes, recipe):
    return _MatrixFree.apply(x, p, value, sizes, recipe)
