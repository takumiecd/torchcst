"""One saved whole-chart state for regular Product and input Strip grids."""

import torch
from torch.autograd.function import once_differentiable


def _contract(t, views, pitch, ranges, *, sizes, recipe, swap):
    import triton as tr

    from . import kernels as k

    b, ni, no, tile, spacing, oi, oo = sizes
    n, origin = (ni, oi) if swap else (no, oo)
    a, owners, splits = views.shape[-1], ranges.shape[1], 4
    y = t.new_empty((b, n))
    partial = t.new_empty((splits, b, n))
    k.contract[(tr.cdiv(b, 16), tr.cdiv(n, 16), splits)](
        t,
        views,
        pitch,
        ranges,
        partial,
        a,
        b,
        n,
        tile,
        spacing,
        origin,
        owners,
        swap,
        recipe.atom_block,
        splits,
        ATOM_MAJOR=getattr(recipe, "atom_group", 1) > 1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    k.reduce_splits[(tr.cdiv(b * n, 256),)](
        partial,
        y,
        b * n,
        splits,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


class _ProfileGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, pitch, value, sizes, recipe):
        import triton as tr

        from ..local_product.preparation import polar_scalars
        from ..profile_product.kernels import prepare
        from . import grouped_kernels as grouped
        from . import kernels as k

        b, ni, no, tile, spacing, oi, oo = sizes
        # Snapshot only once. Replay copies live source/pitch; older VJPs retain
        # the values of their own forward, including the amplitude pullback.
        source = p.contiguous().clone()
        saved_pitch = pitch.clone() if pitch is not None else None
        scalars = polar_scalars(value)
        amplitude_max = scalars[0].clone()
        a, owners = len(p), max(tr.cdiv(ni, 16), tr.cdiv(no, 16))
        packed = p.new_empty((13, a))
        views = p.new_empty((2, 13, a))
        wide = (max(ni, no) + 1) * (a + 1) >= 2**31 - 1
        keys = torch.empty(
            (2, a), device=p.device, dtype=torch.int64 if wide else torch.int32
        )
        order = torch.empty((2, a), device=p.device, dtype=torch.int32)
        inverse = torch.empty(a, device=p.device, dtype=torch.int32)
        ranges = torch.empty((2, owners, 2), device=p.device, dtype=torch.int32)
        h = x.new_empty((b, a))
        if a:
            prep_group = getattr(recipe, "prep_group", 1)
            prep_sites = getattr(recipe, "prep_sites", 32)
            support = getattr(recipe, "preparation", "full") == "support"
            grouped_prep = prep_group > 1 or (tile and support) or prep_sites != 32
            preparation = (
                grouped.prepare
                if grouped_prep
                else (k.prepare_support if support else prepare)
            )
            prep_options = (
                {"GROUP": prep_group, "SUPPORT": support, "PREP_SITES": prep_sites}
                if grouped_prep
                else {}
            )
            preparation[(tr.cdiv(a, prep_group),)](
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
                **prep_options,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if a <= 4096 and getattr(recipe, "sorting", "legacy") == "legacy":
                k.sort_keys[(2,)](
                    packed, keys, a, tr.next_power_of_2(a), wide, num_warps=8
                )
            else:
                k.make_keys[(tr.cdiv(a, 256), 2)](
                    packed, keys, a, 256, wide, num_warps=4
                )
                keys = keys.sort(dim=1).values
            k.copy_views[(tr.cdiv(a, 256), 2)](
                packed, keys, views, order, inverse, a, 256, num_warps=4
            )
            if a <= 4096:
                k.owner_ranges[(2, owners)](
                    views,
                    ranges,
                    a,
                    owners,
                    tr.cdiv(ni, 16),
                    tr.cdiv(no, 16),
                    tr.next_power_of_2(a),
                    num_warps=4,
                )
            else:
                chunks = tr.cdiv(a, 1024)
                highs = torch.empty((2, chunks), device=p.device, dtype=torch.int32)
                k.block_highs[(chunks, 2)](views, highs, a, chunks, 1024, num_warps=4)
                k.chunk_ranges[(2, owners)](
                    views,
                    highs,
                    ranges,
                    a,
                    owners,
                    chunks,
                    tr.next_power_of_2(chunks),
                    1024,
                    num_warps=4,
                )
            atom_group = getattr(recipe, "atom_group", 1)
            input_kernel = grouped.input_h if atom_group > 1 else k.input_h
            group_options = {"GROUP": atom_group} if atom_group > 1 else {}
            input_kernel[(tr.cdiv(a, atom_group),)](
                x,
                views,
                saved_pitch,
                h,
                a,
                b,
                ni,
                tile,
                spacing,
                oi,
                *x.stride(),
                max(16, tr.next_power_of_2(b)),
                8 if atom_group > 1 else 32,
                **group_options,
                num_warps=4,
                enable_fp_fusion=False,
            )
            y = _contract(
                h, views, saved_pitch, ranges, sizes=sizes, recipe=recipe, swap=False
            )
        else:
            y = x.new_zeros((b, no))
        ctx.sizes, ctx.recipe = sizes, recipe
        ctx.save_for_backward(
            x, source, saved_pitch, amplitude_max, views, order, inverse, ranges, h
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import grouped_kernels as grouped
        from . import kernels as k

        x, source, pitch, amplitude_max, views, order, inverse, ranges, h = (
            ctx.saved_tensors
        )
        b, ni, no, tile, spacing, oi, oo = ctx.sizes
        a = len(source)
        dx = dp = None
        need_x, need_p = ctx.needs_input_grad[:2]
        if not a:
            dx = torch.zeros_like(x) if need_x else None
            dp = torch.zeros_like(source) if need_p else None
        elif need_x or need_p:
            g = dy.new_empty((b, a))
            dp = torch.empty_like(source) if need_p else None
            atom_group = getattr(ctx.recipe, "atom_group", 1)
            atom_kernel = grouped.backward_atoms if atom_group > 1 else k.backward_atoms
            group_options = {"GROUP": atom_group} if atom_group > 1 else {}
            atom_kernel[(tr.cdiv(a, atom_group),)](
                x,
                dy,
                views,
                order,
                inverse,
                h,
                g,
                dp,
                source,
                amplitude_max,
                pitch,
                a,
                b,
                ni,
                no,
                tile,
                spacing,
                oi,
                oo,
                *x.stride(),
                *dy.stride(),
                max(16, tr.next_power_of_2(b)),
                8 if atom_group > 1 else 32,
                need_p,
                **group_options,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if need_x:
                dx = _contract(
                    g,
                    views,
                    pitch,
                    ranges,
                    sizes=ctx.sizes,
                    recipe=ctx.recipe,
                    swap=True,
                )
        return dx, dp, None, None, None, None


def grid_product(x, p, value, pitch, sizes, recipe):
    return _ProfileGrid.apply(x, p, pitch, value, sizes, recipe)
