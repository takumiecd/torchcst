"""Small local tiles with production-equivalent polar decoding and task VJP."""

import torch
from torch.autograd.function import once_differentiable

from .preparation import decode, order_atoms, polar_scalars
from .recipe import DEFAULT_RECIPE

# Scalar compiler reports, not CUDA tensors. Never claim on-chip residency if spilled.
COMPILER_REPORTS = {}
# Set only while capturing a separate benchmark diagnostic graph.
DIAGNOSTIC_EVENTS = None


def _stamp(name, edge):
    if DIAGNOSTIC_EVENTS is not None:
        DIAGNOSTIC_EVENTS[name][edge].record()


def _report(name, compiled):
    if name not in COMPILER_REPORTS:
        COMPILER_REPORTS[name] = {
            "registers": getattr(compiled, "n_regs", None),
            "spill_slots": getattr(compiled, "n_spills", None),
            "shared_bytes": getattr(compiled.metadata, "shared", None),
        }


def _sizes(domain, swap=False):
    k, n = domain.input_count, domain.output_count
    oi, oo = domain.input_origin, domain.output_origin
    js, i = domain.input_start, domain.output_start
    return (n, k, oo, oi, i, js) if swap else (k, n, oi, oo, js, i)


def _fused(
    x,
    packed,
    domain,
    recipe,
    swap=False,
    sparse=False,
    hybrid=False,
    h=None,
    support_only=False,
    three_band=False,
    singletons=False,
):
    import triton as tr

    from . import kernels

    if swap:
        _stamp("dx", 0)

    k, n, oi, oo, js, i = _sizes(domain, swap)
    y = x.new_empty((len(x), n))
    grid = (
        (tr.cdiv(len(x), recipe.batch_block), tr.cdiv(n, 16))
        if singletons
        else (tr.cdiv(len(x), recipe.batch_block),)
    )
    compiled = kernels.fused[grid](
        x,
        packed,
        y,
        len(x),
        k,
        n,
        packed.shape[1],
        domain.spacing,
        oi,
        oo,
        js,
        i,
        max(16, tr.next_power_of_2(k)),
        16 if singletons else max(16, tr.next_power_of_2(n)),
        recipe.batch_block,
        recipe.atom_block,
        swap,
        sparse,
        recipe.support_limit,
        hybrid,
        h,
        recipe.rho_upper[1 if three_band else 0],
        support_only,
        THREE_BAND=three_band,
        SINGLETON_FAST=singletons,
        num_warps=8 if max(k, n) > 64 else 4,
        enable_fp_fusion=False,
    )
    _report(
        ("singleton_" if singletons else "")
        + (
            "hybrid_support_"
            if hybrid and sparse
            else "hybrid_"
            if hybrid
            else "support_"
            if sparse
            else ""
        )
        + ("dx" if swap else "fused_forward"),
        compiled,
    )
    if swap:
        _stamp("dx", 1)
    return y


def prepare_metadata(q, domain, *, sparse, scalars, support_bounded=False):
    import triton as tr

    from . import kernels

    _stamp("prepare", 0)

    a = len(q)
    packed = q.new_empty((13 if sparse else 9, a))
    if a:
        prepare = kernels.prepare_support if support_bounded else kernels.prepare
        compiled = prepare[(a,)](
            q,
            packed,
            a,
            domain.input_size,
            domain.output_size,
            domain.spacing,
            domain.input_origin,
            domain.output_origin,
            tr.next_power_of_2(domain.input_size),
            tr.next_power_of_2(domain.output_size),
            sparse,
            bool(scalars),
            scalars,
            num_warps=1 if support_bounded else 4,
            enable_fp_fusion=False,
        )
        _report(
            "prepare_bounded"
            if support_bounded
            else "prepare_polar"
            if scalars
            else "prepare_support"
            if sparse
            else "prepare",
            compiled,
        )
    _stamp("prepare", 1)
    return packed


def tile_layout(packed, domain, recipe):
    """Refresh both execution layouts from current normalized support on device."""
    import triton as tr

    from . import kernels

    a = packed.shape[1]
    views = packed.new_empty((2, 13, a))
    orders = packed.new_empty((2, a), dtype=torch.int32)
    stride = max(tr.cdiv(domain.input_count, 16), tr.cdiv(domain.output_count, 16)) + 5
    offsets = packed.new_empty((2, stride), dtype=torch.int32)
    if a:
        compiled = kernels.pack_tiles[(2,)](
            packed,
            views,
            orders,
            offsets,
            a,
            domain.input_count,
            domain.output_count,
            domain.input_start,
            domain.output_start,
            domain.spacing,
            recipe.rho_upper[1],
            tr.next_power_of_2(a),
            tr.next_power_of_2(stride),
            stride,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("tile_layout", compiled)
    else:
        offsets.zero_()
    return views, orders, offsets


def allocate_owner_index(prototype, atoms, groups, recipe):
    """Shape-only int16 safety check includes every possible persistent slot."""
    slots = ((atoms + 31 * (groups + 4) + 15) // 16) * 16
    dtype = torch.int16 if recipe.index_bits == 16 and slots <= 32768 else torch.int32
    capacity = max(32, ((atoms + 31) // 32) * 32)
    if recipe.release_forward_index:
        ids = tuple(
            prototype.new_empty((groups, capacity), dtype=dtype) for _ in range(2)
        )
    else:
        ids = prototype.new_empty((2, groups, capacity), dtype=dtype)
    offsets = prototype.new_empty((2, groups, 4), dtype=torch.int32)
    return ids, offsets, capacity


def build_owner_index(packed, reverse, domain, recipe, *, physical_views=False):
    import triton as tr

    from . import layout_kernels

    atoms = packed.shape[-1]
    groups = max(tr.cdiv(domain.input_count, 16), tr.cdiv(domain.output_count, 16))
    ids, offsets, capacity = allocate_owner_index(
        packed if physical_views else reverse, atoms, groups, recipe
    )
    compiled = layout_kernels.index_owners[(2, groups)](
        packed,
        reverse,
        ids[0] if recipe.release_forward_index else ids,
        offsets,
        atoms,
        domain.input_count,
        domain.output_count,
        domain.input_start,
        domain.output_start,
        domain.spacing,
        recipe.rho_upper[1],
        max(32, tr.next_power_of_2(atoms)),
        capacity,
        groups,
        ForwardIds=ids[0] if recipe.release_forward_index else None,
        BackwardIds=ids[1] if recipe.release_forward_index else None,
        PHYSICAL_ORDER=physical_views,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("owner_index", compiled)
    return ids, offsets


def ordered_layout(packed, domain, recipe, *, cache=None):
    """Compact current values; per-direction position order and exact owner ranges."""
    import triton as tr

    from . import layout_kernels

    atoms = packed.shape[1]
    groups = max(tr.cdiv(domain.input_count, 16), tr.cdiv(domain.output_count, 16))
    stride = groups + 5
    views = packed.new_empty((2, 13, atoms))
    orders = packed.new_empty((2, atoms), dtype=torch.int32)
    offsets = packed.new_empty((2, stride), dtype=torch.int32)
    ranges = packed.new_empty((2, groups, 6), dtype=torch.int32)
    if atoms:
        compiled = layout_kernels.ordered_views[(2,)](
            packed,
            views,
            orders,
            offsets,
            ranges,
            atoms,
            domain.input_count,
            domain.output_count,
            domain.input_start,
            domain.output_start,
            domain.spacing,
            recipe.rho_upper[1],
            max(32, tr.next_power_of_2(atoms)),
            tr.next_power_of_2(stride),
            stride,
            recipe.order_by_position,
            not recipe.owner_index and not recipe.parallel_owner_ranges,
            recipe.compact_order_key,
            COPY=not recipe.parallel_order_copy,
            VECTOR_RANGES=recipe.vector_owner_ranges,
            OWNER_BLOCK=tr.next_power_of_2(groups),
            CachedKeys=cache.keys if cache is not None else None,
            CachedOrder=cache.order if cache is not None else None,
            CachedOffsets=cache.offsets if cache is not None else None,
            Stats=cache.stats if cache is not None else None,
            CACHE=cache is not None,
            CACHE_LOGICAL=recipe.compact_cached_order,
            CACHE_VALIDATE=recipe.validate_cached_order,
            CACHE_GATHER=recipe.gather_validation_key,
            REPAIR_ROUNDS=recipe.cache_repair_rounds,
            num_warps=recipe.preparation_warps,
            enable_fp_fusion=False,
        )
        _report("ordered_layout", compiled)
        if recipe.parallel_owner_ranges:
            compiled = layout_kernels.ordered_owner_ranges[(2, groups)](
                packed if recipe.parallel_order_copy else views,
                ranges,
                atoms,
                domain.input_count,
                domain.output_count,
                domain.input_start,
                domain.output_start,
                domain.spacing,
                recipe.rho_upper[1],
                max(32, tr.next_power_of_2(atoms)),
                groups,
                Views=views,
                Orders=orders,
                COPY=recipe.parallel_order_copy,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("ordered_owner_ranges", compiled)
    else:
        offsets.zero_()
        ranges.zero_()
    return views, orders, offsets, ranges


def _packed_fused(
    x,
    packed,
    h,
    order,
    offsets,
    domain,
    recipe,
    *,
    swap=False,
    ends=None,
    canonical_atoms=0,
    saved_g=None,
    owner_ids=None,
    owner_offsets=None,
):
    import triton as tr

    from . import kernels

    if swap:
        _stamp("dx", 0)

    k, n, oi, oo, js, i = _sizes(domain, swap)
    splits = recipe.owner_splits
    y = x.new_empty((splits, len(x), n)) if splits > 1 else x.new_empty((len(x), n))
    compiled = kernels.fused_packed[
        (tr.cdiv(len(x), recipe.batch_block), tr.cdiv(n, recipe.output_block), splits)
    ](
        x,
        packed,
        saved_g if saved_g is not None else h,
        order,
        offsets,
        y,
        len(x),
        k,
        n,
        packed.shape[1],
        domain.spacing,
        oi,
        oo,
        js,
        i,
        max(16, tr.next_power_of_2(k)),
        recipe.batch_block,
        recipe.atom_block,
        swap,
        recipe.support_limit,
        recipe.rho_upper[1],
        Ends=ends,
        H_A=packed.shape[1] if recipe.ordered_layout else canonical_atoms,
        SAVED_G=saved_g is not None,
        BAND_DISPATCH=recipe.band_dispatch,
        VECTOR_SUPPORT=recipe.vector_support,
        UNROLL_SUPPORT=recipe.unroll_support,
        BN=recipe.output_block,
        RECOMPUTE_H=recipe.recompute_h,
        OwnerIds=owner_ids,
        OwnerOffsets=owner_offsets,
        INDEX_A=owner_ids.shape[-1] if owner_ids is not None else 0,
        OwnerRanges=owner_offsets
        if recipe.ordered_layout and not recipe.owner_index
        else None,
        PHYSICAL_H=recipe.ordered_layout,
        SPLITS=splits,
        num_warps=recipe.contraction_warps or (8 if max(k, n) > 64 else 4),
        enable_fp_fusion=False,
    )
    _report(
        "packed_dx_saved_g"
        if swap and saved_g is not None
        else "packed_dx"
        if swap
        else "packed_forward",
        compiled,
    )
    if splits > 1:
        result = x.new_empty((len(x), n))
        reduced = kernels.reduce_owner_partials[(tr.cdiv(len(x) * n, 256),)](
            y,
            result,
            len(x) * n,
            splits,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("dx_partial_reduce" if swap else "output_partial_reduce", reduced)
        y = result
    if swap:
        _stamp("dx", 1)
    return y


class _LocalH(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        q,
        domain,
        recipe,
        saved,
        sparse,
        scalars,
        hybrid,
        support_only,
        three_band,
        singletons,
        tile_packed,
        persistent_layout,
    ):
        fused_polar = bool(scalars)
        import triton as tr

        from . import kernels

        a = len(q)
        packed = prepare_metadata(
            q,
            domain,
            sparse=sparse,
            scalars=scalars,
            support_bounded=recipe.support_prepare,
        )
        ends = q.new_empty((0,))
        _stamp("layout", 0)
        owner_ids, owner_offsets = (q.new_empty((0,)),) * 2
        if recipe.ordered_layout:
            if recipe.cached_order:
                from .order_cache import OrderKeyCache

                if not isinstance(persistent_layout, OrderKeyCache):
                    raise ValueError("cached ordering requires model-owned key cache")
                if (
                    persistent_layout.domain != domain
                    or persistent_layout.recipe != recipe
                ):
                    raise ValueError("key cache domain/recipe changed")
                views, orders, offsets, owner_offsets = persistent_layout.refresh(
                    packed
                )
            else:
                views, orders, offsets, owner_offsets = ordered_layout(
                    packed, domain, recipe
                )
            persistent_layout = None
            if recipe.owner_index:
                owner_ids, owner_offsets = build_owner_index(
                    views, None, domain, recipe, physical_views=True
                )
        elif persistent_layout is not None and recipe.fuse_owner_index:
            views, orders, offsets, ends, owner_ids, owner_offsets = (
                persistent_layout.refresh(packed, owner_index=True)
            )
        elif persistent_layout is not None:
            views, orders, offsets, ends = persistent_layout.refresh(packed)
        elif tile_packed:
            views, orders, offsets = tile_layout(packed, domain, recipe)
        else:
            views, orders, offsets = (q.new_empty((0,)),) * 3
        if (
            recipe.owner_index
            and not recipe.fuse_owner_index
            and not recipe.ordered_layout
        ):
            if persistent_layout is None:
                raise ValueError("owner index requires persistent layout")
            owner_ids, owner_offsets = build_owner_index(
                packed, persistent_layout.reverse, domain, recipe
            )
        _stamp("layout", 1)
        # Fixed capacity keeps graph replay independent of a changing wide count.
        # Hybrid writes/reads only wide lanes; compact capacity is future work.
        _stamp("h", 0)
        keep_h = (saved or hybrid) and not recipe.recompute_h
        h = q.new_empty((len(x), a)) if keep_h else q.new_empty((0,))
        if keep_h and a:
            compiled = kernels.save_h[
                (tr.cdiv(len(x), recipe.batch_block), tr.cdiv(a, recipe.atom_block))
            ](
                x,
                views[0] if recipe.ordered_layout else packed,
                h,
                len(x),
                domain.input_count,
                a,
                domain.spacing,
                domain.input_origin,
                domain.input_start,
                max(16, tr.next_power_of_2(domain.input_count)),
                recipe.batch_block,
                recipe.atom_block,
                hybrid,
                recipe.rho_upper[1 if three_band else 0],
                support_only,
                THREE_BAND=three_band,
                SINGLETON_FAST=singletons,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("hybrid_save_h" if hybrid else "save_h", compiled)
        _stamp("h", 1)
        _stamp("output", 0)
        if tile_packed:
            y = _packed_fused(
                x,
                views[0],
                h,
                orders[0],
                offsets[0],
                domain,
                recipe,
                ends=ends[0] if persistent_layout is not None else None,
                canonical_atoms=a if persistent_layout is not None else 0,
                owner_ids=owner_ids[0] if recipe.owner_index else None,
                owner_offsets=owner_offsets[0]
                if recipe.owner_index or recipe.ordered_layout
                else None,
            )
        elif saved:
            y = x.new_empty((len(x), domain.output_count))
            compiled = kernels.from_h[(tr.cdiv(len(x), recipe.batch_block),)](
                h,
                packed,
                y,
                len(x),
                domain.output_count,
                a,
                domain.spacing,
                domain.output_origin,
                domain.output_start,
                max(16, tr.next_power_of_2(domain.output_count)),
                recipe.batch_block,
                recipe.atom_block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("from_h", compiled)
        else:
            y = _fused(
                x,
                packed,
                domain,
                recipe,
                sparse=sparse,
                hybrid=hybrid,
                h=h,
                support_only=support_only,
                three_band=three_band,
                singletons=singletons,
            )
        _stamp("output", 1)
        if recipe.release_forward_h and (
            not recipe.recompute_param_h or recipe.save_g or saved
        ):
            raise ValueError("forward-only H requires recomputed parameter VJP and dX")
        ctx.save_for_backward(
            x,
            views[0] if tile_packed else packed,
            q.new_empty((0,)) if recipe.release_forward_h else h,
            views,
            orders,
            offsets,
            ends,
            owner_ids[1] if recipe.release_forward_index else owner_ids,
            owner_offsets[1] if recipe.release_forward_index else owner_offsets,
            q if fused_polar else q.new_empty((0,)),
            *scalars,
        )
        ctx.settings = (
            domain,
            recipe,
            saved,
            sparse,
            fused_polar,
            hybrid,
            support_only,
            three_band,
            singletons,
            tile_packed,
            persistent_layout is not None,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        import triton as tr

        from . import kernels

        (
            x,
            packed,
            h,
            views,
            orders,
            offsets,
            ends,
            owner_ids,
            owner_offsets,
            source,
            *scalars,
        ) = ctx.saved_tensors
        (
            domain,
            recipe,
            saved,
            sparse,
            fused_polar,
            hybrid,
            support_only,
            three_band,
            singletons,
            tile_packed,
            persistent,
        ) = ctx.settings
        dx_ids = (
            owner_ids
            if recipe.release_forward_index
            else owner_ids[1]
            if recipe.owner_index
            else None
        )
        dx_offsets = (
            owner_offsets[1]
            if recipe.ordered_layout and not recipe.owner_index
            else owner_offsets
            if recipe.release_forward_index
            else owner_offsets[1]
            if recipe.owner_index
            else None
        )
        dy = dy.contiguous()
        dx = None
        use_g = (
            recipe.save_g
            and tile_packed
            and ctx.needs_input_grad[0]
            and ctx.needs_input_grad[1]
        )
        g = x.new_empty((len(x), len(source))) if use_g else None
        if ctx.needs_input_grad[0] and tile_packed and not use_g:
            dx = _packed_fused(
                dy,
                views[1],
                h,
                orders[1],
                offsets[1],
                domain,
                recipe,
                swap=True,
                ends=ends[1] if persistent else None,
                canonical_atoms=len(source) if persistent else 0,
                owner_ids=dx_ids,
                owner_offsets=dx_offsets,
            )
        elif ctx.needs_input_grad[0] and not use_g:
            dx = _fused(
                dy,
                packed,
                domain,
                recipe,
                swap=True,
                sparse=sparse and not hybrid,
                support_only=support_only,
                singletons=singletons,
            )
        dq = None
        if ctx.needs_input_grad[1]:
            _stamp("parameters", 0)
            a = packed.shape[1]
            canonical_atoms = len(source) if persistent else a
            dq = packed.new_empty((canonical_atoms, 4))
            if a:
                compiled = kernels.param_vjp[(tr.cdiv(a, recipe.atom_block),)](
                    x,
                    dy,
                    packed,
                    h,
                    dq,
                    len(x),
                    domain.input_count,
                    domain.output_count,
                    a,
                    domain.spacing,
                    domain.input_origin,
                    domain.output_origin,
                    domain.input_start,
                    domain.output_start,
                    max(16, tr.next_power_of_2(domain.input_count)),
                    max(16, tr.next_power_of_2(domain.output_count)),
                    16
                    if max(domain.input_count, domain.output_count) > 64 or use_g
                    else max(16, tr.next_power_of_2(len(x))),
                    recipe.atom_block,
                    saved,
                    sparse,
                    recipe.support_limit,
                    fused_polar,
                    source if fused_polar else None,
                    scalars[0] if fused_polar else None,
                    hybrid and not recipe.recompute_param_h,
                    recipe.rho_upper[1 if three_band else 0],
                    support_only,
                    THREE_BAND=three_band,
                    SINGLETON_FAST=singletons,
                    Order=orders[0] if tile_packed else None,
                    H_A=canonical_atoms if persistent else 0,
                    G=g,
                    SAVE_G=use_g,
                    num_warps=recipe.parameter_warps
                    or (8 if max(domain.input_count, domain.output_count) > 64 else 4),
                    enable_fp_fusion=False,
                )
                _report(
                    "singleton_param"
                    if singletons
                    else "hybrid_support_param"
                    if hybrid and sparse
                    else "hybrid_param"
                    if hybrid
                    else "polar_param"
                    if fused_polar
                    else "support_param"
                    if sparse
                    else ("param_saved" if saved else "param_recomputed"),
                    compiled,
                )
            _stamp("parameters", 1)
        if use_g:
            dx = _packed_fused(
                dy,
                views[1],
                h,
                orders[1],
                offsets[1],
                domain,
                recipe,
                swap=True,
                ends=ends[1] if persistent else None,
                canonical_atoms=len(source),
                saved_g=g,
                owner_ids=dx_ids,
                owner_offsets=dx_offsets,
            )
        return dx, dq, None, None, None, None, None, None, None, None, None, None, None


def local_h(
    x,
    p,
    value,
    domain,
    *,
    saved=False,
    sparse=False,
    fused_polar=False,
    hybrid=False,
    support_only=False,
    three_band=False,
    singletons=False,
    tile_packed=False,
    persistent_layout=None,
    recipe=DEFAULT_RECIPE,
):
    """Y_local from X_local, normalized full-domain profiles, and polar atoms.

    Call validate_state once at configuration. Bounds stay shared during updates.
    No outer GEMM scheduling or production Strip/Torus dispatch is added here.
    """
    if recipe.recompute_h and (not tile_packed or not hybrid):
        raise ValueError("on-chip H requires the packed hybrid execution route")
    if support_only and (not sparse or hybrid):
        raise ValueError("support-only requires sparse and no hybrid classification")
    if sparse and saved and not support_only:
        raise ValueError("initial support route recomputes local H")
    if hybrid and (saved or recipe.pack):
        raise ValueError(
            "hybrid requires canonical order and no whole-call saved route"
        )
    if three_band and (
        not hybrid
        or not sparse
        or len(recipe.rho_upper) < 3
        or recipe.rho_upper[0] != 1
    ):
        raise ValueError(
            "three-band requires sparse hybrid with boundaries [1, mid, ...]"
        )
    if persistent_layout is not None and (
        not tile_packed
        or persistent_layout.domain != domain
        or persistent_layout.recipe != recipe
    ):
        raise ValueError("persistent layout configuration differs from execution")
    if tile_packed and (not singletons or not fused_polar):
        raise ValueError("tile packing requires fused polar singleton hybrid")
    if singletons and not three_band:
        raise ValueError("singleton split requires the three-band hybrid")
    if (
        not x.is_cuda
        or x.dtype != torch.float32
        or p.dtype != torch.float32
        or p.device != x.device
        or x.ndim != 2
        or p.ndim != 2
        or p.shape[1] != 4
        or not 1 <= len(x) <= 64
        or x.shape[1] != domain.input_count
    ):
        raise ValueError("requires CUDA FP32 local X[B<=64,K] and polar P[A,4]")
    if fused_polar:
        if recipe.pack:
            raise ValueError("initial fused polar route uses canonical atom order")
        q, scalars = p, polar_scalars(value)
    else:
        q, scalars = decode(value, p), ()
        order = order_atoms(q, domain, recipe)
        q = q.index_select(0, order)
    with torch.cuda.device(x.device):
        return _LocalH.apply(
            x.contiguous(),
            q.contiguous(),
            domain,
            recipe,
            saved,
            sparse,
            scalars,
            hybrid,
            support_only,
            three_band,
            singletons,
            tile_packed,
            persistent_layout,
        )
