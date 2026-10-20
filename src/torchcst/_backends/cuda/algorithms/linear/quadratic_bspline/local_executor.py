"""Immutable local-moment snapshots and complete first-order autograd."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import local_kernels as lk
from .executor import _prepare, _report, _stamp
from .native_runtime import backward as native_backward
from .native_runtime import forward as native_forward
from .stream_layout import make_layout, placement


def _groups(n, recipe):
    sizes = tuple(
        1 << k for k in range(getattr(recipe, "min_block_log", 3), n.bit_length())
    )
    count = recipe.bank_group_levels
    return tuple(sizes[k : k + count] for k in range(0, len(sizes), count))


def _points(n, recipe):
    minimum = getattr(recipe, "min_block_log", 3)
    return (n.bit_length() - minimum) * n + 2 * (n >> minimum) - 1


def _blocks(n, recipe):
    return 2 * (n >> getattr(recipe, "min_block_log", 3)) - 1


def _h_launch(direction, atoms, recipe):
    module = lk
    programs = tr.cdiv(atoms, recipe.axis_atoms)
    if getattr(recipe, "input_groups", False):
        from . import grouped_kernels

        module = grouped_kernels
    if getattr(recipe, "transposed_h", False):
        from . import transposed_kernels

        module = transposed_kernels
    if getattr(recipe, "gluon_h", False):
        from . import gluon_kernels

        module = gluon_kernels
    if getattr(recipe, "unified_native", False):
        from . import unified_kernels

        module = unified_kernels
    cap = getattr(recipe, "h_programs", 0)
    if cap:
        from . import local_persistent_kernels

        module = local_persistent_kernels
        programs = min(programs, cap)
    return getattr(module, direction)[(programs,)]


def _group_tables(pos, count, atoms, sites, recipe):
    if not getattr(recipe, "input_groups", False):
        return ()
    from .input_groups import prepare

    _stamp("input_group_prepare", 0)
    table = prepare(pos, count, atoms, sites, recipe.axis_atoms, recipe.min_block_log)
    # Counts are construction scratch; queries need only these four snapshots.
    result = (table.ends, table.ids, table.chunk_ends, table.mapping)
    _stamp("input_group_prepare", 1)
    return result


def _h_args(tables, recipe):
    if not tables:
        return (
            {"LANE_ATOMS": recipe.lane_atoms}
            if getattr(recipe, "gluon_h", False)
            else {}
        )
    ends, ids, chunks, mapping = tables
    return {
        "GroupEnds": ends,
        "GroupIDs": ids,
        "ChunkEnds": chunks,
        "ChunkMap": mapping,
        "CAP": len(ends),
        "SHARED_INPUT": recipe.shared_input,
        "REDUCE_DX": recipe.group_dx,
    }


def _native_h_args(x, dy, destination, sizes, recipe):
    if not getattr(recipe, "unified_native", False):
        return {}
    result = {
        "RawX": x,
        "RawDY": dy,
        "NativeOut": destination,
        "NO": sizes[0],
        "NI": sizes[1],
    }
    if isinstance(destination, tuple):
        cold, hot = destination
        result.update(
            NativeOut=cold,
            NativeHotOut=hot,
            HOT_SHARDS=recipe.native_hot_shards,
            HOT_STRIDE=hot.stride(0),
        )
    if getattr(recipe, "native_pointmajor", False):
        result.update(
            RawX=x.t().contiguous(),
            RawDY=None if dy is None else dy.t().contiguous(),
            NATIVE_POINT_MAJOR=True,
            HOT_ROWS=destination[1].stride(1) if destination is not None else 0,
        )
    return result


def _native_part(destination, start, actual, recipe):
    if destination is None:
        return None
    if isinstance(destination, tuple):
        cold, hot = destination
        selected = (
            hot[:, :, :actual]
            if getattr(recipe, "native_pointmajor", False)
            else hot[:, :actual]
        )
        return cold[start : start + actual], selected
    return destination[start : start + actual]


def _finish_native(output, destination, label, recipe):
    if destination is None:
        return
    if not isinstance(destination, tuple):
        output.add_(destination)
        return
    from . import native_shard_kernels as ns

    cold, hot = destination
    _stamp(label, 0)
    compiled = ns.finish[(tr.cdiv(output.numel(), 256),)](
        output,
        cold,
        hot,
        output.numel(),
        hot.stride(0),
        hot.shape[0],
        256,
        POINT_MAJOR=getattr(recipe, "native_pointmajor", False),
        ROWS=hot.stride(1) if getattr(recipe, "native_pointmajor", False) else 0,
        SITES=output.shape[1],
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report(label, compiled)
    _stamp(label, 1)


def _cache_args(pos, co, active, recipe):
    if not getattr(recipe, "native_cache", False):
        return None
    return (pos, co, active) if getattr(recipe, "unified_native", False) else (pos, co)


def _tile_bundle(n, sizes, recipe):
    if not hasattr(recipe, "bank_tile_blocks") or sizes[0] >= 256:
        return 1
    requested = recipe.bank_tile_blocks if sizes[0] < 32 else recipe.bank_mid_blocks
    return min(requested, n // sizes[0])


def _bank(x, n, recipe, label, active, dim, ml, buffer=None):
    bank = (
        buffer
        if buffer is not None
        else x.new_empty(
            (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0))
        )
    )
    pad = bank.shape[2] - len(x)
    _stamp(label, 0)
    for k, sizes in enumerate(_groups(n, recipe)):
        tiles = _tile_bundle(n, sizes, recipe)
        kernel = lk.bank
        if tiles > 1:
            from . import bundled_bank_kernels as bk

            kernel = bk.bank
        compiled = kernel[
            (
                sum(tr.cdiv(n // s, tiles) for s in sizes),
                tr.cdiv(len(x), recipe.bank_batch),
            )
        ](
            x,
            bank,
            n,
            len(x),
            _points(n, recipe),
            sizes[0],
            len(sizes),
            sizes[-1],
            recipe.bank_batch,
            active if getattr(recipe, "active_levels", False) else None,
            ml,
            dim,
            getattr(recipe, "active_levels", False),
            pad,
            getattr(recipe, "min_block_log", 3),
            **({"TB": tiles} if tiles > 1 else {}),
            MASK_MODE=getattr(recipe, "fused_levels", False),
            num_warps=recipe.bank_warps,
            enable_fp_fusion=False,
        )
        _report(label + f"_{k}", compiled)
    _stamp(label, 1)
    return bank


def _restore(delta, border, y, n, recipe, label, active, dim, ml):
    _stamp(label, 0)
    for k, sizes in enumerate(_groups(n, recipe)):
        tiles = _tile_bundle(n, sizes, recipe)
        kernel = lk.restore
        if tiles > 1:
            from . import bundled_bank_kernels as bk

            kernel = bk.restore
        compiled = kernel[
            (
                sum(tr.cdiv(n // s, tiles) for s in sizes),
                tr.cdiv(len(y), recipe.bank_batch),
            )
        ](
            delta,
            y,
            n,
            len(y),
            _points(n, recipe),
            sizes[0],
            len(sizes),
            sizes[-1],
            recipe.bank_batch,
            active if getattr(recipe, "active_levels", False) else None,
            ml,
            dim,
            getattr(recipe, "active_levels", False),
            border,
            (getattr(recipe, "boundary_shards", 0) if border is not None else 0),
            delta.shape[2] - len(y),
            getattr(recipe, "min_block_log", 3),
            **({"TB": tiles} if tiles > 1 else {}),
            MASK_MODE=getattr(recipe, "fused_levels", False),
            num_warps=recipe.bank_warps,
            enable_fp_fusion=False,
        )
        _report(label + f"_{k}", compiled)
    _stamp(label, 1)
    return y


def _delta(x, n, recipe, active, dim, ml, buffers=None, label=None):
    label = label or ("delta_output" if dim == 0 else "delta_input")
    _stamp(label, 0)
    if buffers is None:
        delta = x.new_empty(
            (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0))
        )
        shards = getattr(recipe, "boundary_shards", 0)
        border = (
            x.new_empty((3, _blocks(n, recipe), shards, delta.shape[2]))
            if shards
            else None
        )
    else:
        delta, border = buffers
    pad = delta.shape[2] - len(x)
    if getattr(recipe, "active_levels", False) and getattr(recipe, "zero_active", True):
        for k, sizes in enumerate(_groups(n, recipe)):
            compiled = lk.zero[
                (sum(n // v for v in sizes), tr.cdiv(len(x), recipe.bank_batch))
            ](
                delta,
                active,
                n,
                len(x),
                _points(n, recipe),
                sizes[0],
                len(sizes),
                sizes[-1],
                recipe.bank_batch,
                ml,
                dim,
                pad,
                getattr(recipe, "min_block_log", 3),
                MASK_MODE=getattr(recipe, "fused_levels", False),
                num_warps=recipe.bank_warps,
                enable_fp_fusion=False,
            )
            _report(label + f"_{k}", compiled)
    else:
        delta.zero_()
    if border is not None:
        border.zero_()
    _stamp(label, 1)
    return delta, border


def _source_order(order, recipe):
    return order if getattr(recipe, "canonical_meta", False) else None


def _frame(x, p, value, chart, recipe, moments, routing, sizes, build_border=True):
    a = len(p)
    meta, origin, spacing, _source = _prepare(p, value, chart, recipe, moments)
    _stamp("group", 0)
    order, count = placement(meta, origin, spacing, sizes, recipe, routing)
    _stamp("group", 1)
    _stamp("pack", 0)
    if not getattr(recipe, "canonical_meta", False):
        meta = meta.index_select(0, order)
    _stamp("pack", 1)
    _stamp("coeff_prepare", 0)
    endpoints = getattr(recipe, "skip_endpoints", False)
    injection = (
        x.new_empty((2, a), dtype=torch.int16)
        if endpoints
        else x.new_empty(0, dtype=torch.int16)
    )
    pos = x.new_empty((2, 6, a), dtype=torch.int32)
    layout = getattr(recipe, "coefficient_layout", "soa")
    co = x.new_empty(
        (2, 3, 6, a) if layout == "soa" else (2, a, 6, 4 if layout == "aos4" else 3)
    )
    half = x.new_empty((2, a))
    shards = getattr(recipe, "boundary_shards", 0) if build_border else 0
    bp = (
        x.new_empty((2, 2, a), dtype=torch.int32)
        if shards
        else x.new_empty(0, dtype=torch.int32)
    )
    fused_levels = getattr(recipe, "fused_levels", False)
    active = x.new_zeros(2, dtype=torch.int32) if fused_levels else None
    compiled = lk.prepare[(tr.cdiv(a, 128),)](
        meta,
        pos,
        co,
        half,
        count,
        origin,
        spacing,
        a,
        *sizes,
        128,
        recipe.reciprocal,
        recipe.prepare_float32,
        bp if shards else None,
        shards,
        injection if endpoints else None,
        endpoints,
        getattr(recipe, "min_block_log", 3),
        Order=_source_order(order, recipe),
        LAYOUT=getattr(recipe, "coefficient_layout", "soa"),
        COORDINATES=getattr(recipe, "coefficient_coordinates", False),
        ActiveMask=active,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("coeff_prepare", compiled)
    _stamp("coeff_prepare", 1)
    if getattr(recipe, "native_cache", False):
        from . import native_cache_kernels as nc

        _stamp("native_cache_prepare", 0)
        compiled = nc.prepare[(tr.cdiv(a, 128),)](
            meta,
            pos,
            co,
            count,
            _source_order(order, recipe),
            origin,
            spacing,
            a,
            *sizes,
            recipe.reciprocal,
            GROUP_ATOMS=recipe.axis_atoms
            if getattr(recipe, "unified_native", False)
            else 0,
            ActiveMask=active if getattr(recipe, "unified_native", False) else None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("native_cache_prepare", compiled)
        _stamp("native_cache_prepare", 1)
    ml = max(sizes).bit_length() - getattr(recipe, "min_block_log", 3)
    _stamp("active_levels", 0)
    if getattr(recipe, "active_levels", False) and not fused_levels:
        active = x.new_empty((2, ml), dtype=torch.int32)
        compiled = lk.classify_levels[(2, ml)](
            pos,
            count,
            active,
            a,
            *sizes,
            ml,
            512,
            getattr(recipe, "min_block_log", 3),
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("active_levels", compiled)
    elif not fused_levels:
        active = x.new_empty(0, dtype=torch.int32)
    _stamp("active_levels", 1)
    return (
        meta,
        origin,
        spacing,
        order,
        count,
        pos,
        co,
        half,
        active,
        bp,
        injection,
        ml,
        shards,
        endpoints,
    )


class _Local(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, routing):
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        a, b = len(p), len(x)
        ctx.recipe, ctx.sizes = recipe, sizes
        ctx.mode = recipe.save_h if ctx.needs_input_grad[1] else "none"
        ctx.retain_input_bank = (
            getattr(recipe, "save_input_bank", False) and ctx.needs_input_grad[1]
        )
        ctx.empty = not a
        if not a:
            ctx.save_for_backward(x)
            return x.new_zeros((b, sizes[0]))
        (
            meta,
            origin,
            spacing,
            order,
            count,
            pos,
            co,
            half,
            active,
            bp,
            injection,
            ml,
            shards,
            endpoints,
        ) = _frame(x, p, value, chart, recipe, moments, routing, sizes)
        groups = _group_tables(pos, count, a, sizes[1], recipe)
        y = x.new_zeros((b, sizes[0]))
        native_y = native_forward(
            x,
            y,
            meta,
            count,
            origin,
            spacing,
            sizes,
            recipe,
            order=_source_order(order, recipe),
            cache=_cache_args(pos, co, active, recipe),
        )
        bank = _bank(x, sizes[1], recipe, "bank_input", active, 1, ml)
        delta, border = _delta(x, sizes[0], recipe, active, 0, ml)
        h = x.new_empty((a, b)) if ctx.mode != "none" else x.new_empty(0)
        hp = x.new_empty((a, b)) if ctx.mode == "hp" else x.new_empty(0)
        _stamp("stream_forward", 0)
        compiled = _h_launch("forward", a, recipe)(
            bank,
            meta,
            pos,
            co,
            half,
            h,
            hp,
            delta,
            count,
            a,
            b,
            _points(sizes[1], recipe),
            _points(sizes[0], recipe),
            recipe.axis_atoms,
            tr.next_power_of_2(b),
            ctx.mode,
            border,
            bp if shards else None,
            shards,
            _blocks(sizes[0], recipe),
            injection if endpoints else None,
            endpoints,
            getattr(recipe, "shard_origin", False),
            getattr(recipe, "bank_padding", 0),
            Order=_source_order(order, recipe),
            LAYOUT=getattr(recipe, "coefficient_layout", "soa"),
            **_h_args(groups, recipe),
            **_native_h_args(x, None, native_y, sizes, recipe),
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _report("stream_forward", compiled)
        _stamp("stream_forward", 1)
        _finish_native(y, native_y, "native_finish_forward", recipe)
        y = _restore(delta, border, y, sizes[0], recipe, "scan_output", active, 0, ml)
        ctx.save_for_backward(
            meta,
            x,
            origin,
            spacing,
            order,
            count,
            pos,
            co,
            half,
            h,
            hp,
            active,
            bp,
            injection,
            bank if ctx.retain_input_bank else h,
            *groups,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            (x,) = ctx.saved_tensors
            return (
                torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
                x.new_empty((0, 4)) if ctx.needs_input_grad[1] else None,
                None,
                None,
                None,
                None,
                None,
            )
        (
            meta,
            x,
            origin,
            spacing,
            order,
            count,
            pos,
            co,
            half,
            h,
            hp,
            active,
            bp,
            injection,
            saved_bank,
        ) = ctx.saved_tensors[:15]
        groups = ctx.saved_tensors[15:]
        dy = dy.contiguous()
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b = len(meta), len(x)
        ml = max(sizes).bit_length() - getattr(recipe, "min_block_log", 3)
        shards = getattr(recipe, "boundary_shards", 0)
        endpoints = getattr(recipe, "skip_endpoints", False)
        need_dx, need_dp = ctx.needs_input_grad[:2]
        native_result = native_backward(
            x,
            dy,
            meta,
            count,
            origin,
            spacing,
            sizes,
            recipe,
            need_dx,
            need_dp,
            order=_source_order(order, recipe),
            cache=_cache_args(pos, co, active, recipe),
        )
        if getattr(recipe, "unified_native", False):
            dx, dp, native_dx = native_result
        else:
            dx, dp = native_result
            native_dx = None
        if need_dp and ctx.mode != "hp":
            if ctx.retain_input_bank:
                input_bank = saved_bank
                _stamp("bank_recompute", 0)
                _stamp("bank_recompute", 1)
            else:
                input_bank = _bank(x, sizes[1], recipe, "bank_recompute", active, 1, ml)
        else:
            input_bank = None
            _stamp("bank_recompute", 0)
            _stamp("bank_recompute", 1)
        adjoint_bank = _bank(dy, sizes[0], recipe, "bank_adjoint", active, 0, ml)
        delta, border = _delta(x, sizes[1], recipe, active, 1, ml)
        _stamp("stream_backward", 0)
        compiled = _h_launch("backward", a, recipe)(
            input_bank,
            adjoint_bank,
            meta,
            pos,
            co,
            half,
            h,
            hp,
            delta,
            dp,
            count,
            a,
            b,
            _points(sizes[1], recipe),
            _points(sizes[0], recipe),
            recipe.axis_atoms,
            tr.next_power_of_2(b),
            ctx.mode,
            need_dx,
            need_dp,
            recipe.contract_float32,
            border,
            bp if shards else None,
            shards,
            _blocks(sizes[1], recipe),
            injection if endpoints else None,
            endpoints,
            getattr(recipe, "shard_origin", False),
            getattr(recipe, "bank_padding", 0),
            Order=_source_order(order, recipe),
            LAYOUT=getattr(recipe, "coefficient_layout", "soa"),
            **_h_args(groups, recipe),
            **_native_h_args(x, dy, native_dx, sizes, recipe),
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _report("stream_backward", compiled)
        _stamp("stream_backward", 1)
        if need_dx:
            _finish_native(dx, native_dx, "native_finish_backward", recipe)
            dx = _restore(
                delta, border, dx, sizes[1], recipe, "scan_input", active, 1, ml
            )
        else:
            _stamp("scan_input", 0)
            _stamp("scan_input", 1)
        _stamp("unpack", 0)
        if need_dp and not getattr(recipe, "canonical_meta", False):
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return dx, dp if need_dp else None, None, None, None, None, None


def apply(x, p, value, chart, recipe, bases, moments, buffers):
    sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
    key = p.device, len(p), chart.grid_shape, recipe
    if buffers.get("stream_layout_key") != key:
        buffers["stream_layout"] = make_layout(len(p), sizes, recipe, p.device)
        buffers["stream_layout_key"] = key
    if recipe.norm_float32:
        if "local_norm_moments" not in buffers:
            count = torch.arange(moments.shape[1], device=p.device, dtype=torch.float64)
            safe = count.clamp_min(1)
            m2, m4 = moments
            buffers["local_norm_moments"] = torch.stack(
                (
                    m2,
                    m4,
                    m2 / safe,
                    (m4 - m2.square() / safe).clamp_min(0),
                    torch.where(count > 0, safe.reciprocal(), 0),
                )
            )
        moments = buffers["local_norm_moments"]
    function = _Local
    if getattr(recipe, "owner_events", 0):
        from .owner_executor import _Owner

        function = _Owner
    elif getattr(recipe, "batch_chunk", 0) and len(x) > recipe.batch_chunk:
        from .chunk_executor import _Chunked

        function = _Chunked
    return function.apply(
        x.contiguous(), p, value, chart, recipe, moments, buffers["stream_layout"]
    )
