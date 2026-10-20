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


def _h_batch(batch, recipe):
    return max(
        tr.next_power_of_2(batch), 32 if getattr(recipe, "batch_values", 1) == 4 else 1
    )


def _h_args(tables, recipe, tensor=None, geometry=None):
    if not tables:
        result = (
            {"LANE_ATOMS": recipe.lane_atoms}
            if getattr(recipe, "gluon_h", False)
            else {}
        )
        if getattr(recipe, "batch_values", 1) == 4:
            major, _ = torch.cuda.get_device_capability(tensor.device)
            result.update(
                BATCH_VALUES=4, VECTOR_RED=recipe.vector_atomic and major >= 9
            )
        if getattr(recipe, "looped_query", False):
            result["LOOPED_QUERY"] = True
        if getattr(recipe, "compact_c2", False):
            result["COMPACT_C2"] = True
        if getattr(recipe, "packed_bank", False):
            result["PACKED_BANK"] = True
        if getattr(recipe, "bit_split_query", False):
            result["BIT_SPLIT_QUERY"] = True
        if getattr(recipe, "paired_query", False):
            result["PAIRED_QUERY"] = True
        if getattr(recipe, "integer_query", False):
            origin, spacing, sizes = geometry
            result.update(
                INTEGER=True,
                QueryOrigin=origin,
                QuerySpacing=spacing,
                I_NO=sizes[0],
                I_NI=sizes[1],
            )
        return result
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
    if getattr(recipe, "integer_query", False):
        return {"RawInput": x, "RawAdjoint": dy}
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


def _bank_dtype(x, recipe):
    return torch.float64 if getattr(recipe, "precise_bank", False) else x.dtype


def _delta_dtype(x, recipe):
    return torch.float64 if getattr(recipe, "precise_local", False) else x.dtype


def _query_buffer(x, n, batch, recipe):
    if getattr(recipe, "integer_query", False):
        return x.new_empty((n + 1, batch), dtype=torch.int64)
    return x.new_empty((3, _points(n, recipe), batch), dtype=_bank_dtype(x, recipe))


def _bank(x, n, recipe, label, active, dim, ml, buffer=None, zero_buffers=None):
    if getattr(recipe, "integer_query", False):
        from . import integer_query_kernels as iq

        bank = (
            buffer
            if buffer is not None
            else _query_buffer(
                x, n, len(x) + getattr(recipe, "bank_padding", 0), recipe
            )
        )
        delta, border = zero_buffers if zero_buffers is not None else (None, None)
        zero_count = (delta.numel() if delta is not None else 0) + (
            border.numel() if border is not None else 0
        )
        _stamp(label, 0)
        compiled = iq.bank[(bank.shape[1] + tr.cdiv(zero_count, 1024),)](
            x,
            bank,
            n,
            len(x),
            bank.shape[1],
            delta,
            border,
            delta.numel() if delta is not None else 0,
            border.numel() if border is not None else 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report(label + "_integer", compiled)
        _stamp(label, 1)
        return bank
    bank = (
        buffer
        if buffer is not None
        else _query_buffer(x, n, len(x) + getattr(recipe, "bank_padding", 0), recipe)
    )
    pad = bank.shape[2] - len(x)
    _stamp(label, 0)
    for k, sizes in enumerate(_groups(n, recipe)):
        tiles = _tile_bundle(n, sizes, recipe)
        kernel = lk.bank
        if tiles > 1:
            from . import bundled_bank_kernels as bk

            kernel = bk.bank
        grid0 = sum(tr.cdiv(n // s, tiles) for s in sizes)
        grid1 = tr.cdiv(len(x), recipe.bank_batch)
        extra = {}
        if zero_buffers is not None and k == 0:
            from .bank_zero_kernels import bank_and_zero

            delta, border = zero_buffers
            zero_count = delta.numel() + (border.numel() if border is not None else 0)
            kernel = bank_and_zero
            grid = (grid0 * grid1 + tr.cdiv(zero_count, 1024),)
            extra = {
                "Delta": delta,
                "Border": border,
                "GRID0": grid0,
                "GRID1": grid1,
                "DELTA_ELEMENTS": delta.numel(),
                "BORDER_ELEMENTS": border.numel() if border is not None else 0,
            }
        else:
            grid = (grid0, grid1)
        compiled = kernel[grid](
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
            **({"TB": tiles} if tiles > 1 or extra else {}),
            MASK_MODE=getattr(recipe, "fused_levels", False),
            **({"PACKED_BANK": True} if getattr(recipe, "packed_bank", False) else {}),
            **extra,
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


def _delta_buffers(x, n, recipe, buffers=None):
    if buffers is None:
        delta = x.new_empty(
            (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0)),
            dtype=_delta_dtype(x, recipe),
        )
        shards = getattr(recipe, "boundary_shards", 0)
        border = (
            x.new_empty(
                (3, _blocks(n, recipe), shards, delta.shape[2]),
                dtype=_delta_dtype(x, recipe),
            )
            if shards
            else None
        )
    else:
        delta, border = buffers
    return delta, border


def _bank_and_delta(
    x,
    bank_n,
    delta_n,
    recipe,
    bank_label,
    delta_label,
    active,
    bank_dim,
    delta_dim,
    ml,
    bank_buffer=None,
    delta_buffers=None,
):
    if getattr(recipe, "fuse_bank_zero", False):
        delta, border = _delta_buffers(x, delta_n, recipe, delta_buffers)
        bank = _bank(
            x,
            bank_n,
            recipe,
            bank_label,
            active,
            bank_dim,
            ml,
            bank_buffer,
            (delta, border),
        )
        # Zero CTAs are timed inside bank_label, not a separate zero kernel.
        _stamp(delta_label, 0)
        _stamp(delta_label, 1)
    else:
        bank = _bank(x, bank_n, recipe, bank_label, active, bank_dim, ml, bank_buffer)
        delta, border = _delta(
            x, delta_n, recipe, active, delta_dim, ml, delta_buffers, delta_label
        )
    return bank, delta, border


def _delta(x, n, recipe, active, dim, ml, buffers=None, label=None):
    label = label or ("delta_output" if dim == 0 else "delta_input")
    _stamp(label, 0)
    delta, border = _delta_buffers(x, n, recipe, buffers)
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


def _frame(
    x,
    p,
    value,
    chart,
    recipe,
    moments,
    routing,
    sizes,
    build_border=True,
    numeric_out=None,
):
    a = len(p)
    endpoints = getattr(recipe, "skip_endpoints", False)
    shards = getattr(recipe, "boundary_shards", 0) if build_border else 0
    fused_levels = getattr(recipe, "fused_levels", False)
    frame = None
    if getattr(recipe, "frame_arena", False):
        from .frame_buffers import make_frame, numeric_layout

        numeric_elements = 0
        if getattr(recipe, "whole_frame", False):
            _, _, numeric_elements = numeric_layout(
                input_points=_points(sizes[1], recipe),
                output_points=_points(sizes[0], recipe),
                blocks=max(_blocks(n, recipe) for n in sizes),
                batch=len(x),
                shards=shards,
            )
        frame = make_frame(
            p,
            a,
            shards=shards,
            endpoints=endpoints,
            active=fused_levels,
            numeric_elements=numeric_elements,
            compact_positions=getattr(recipe, "compact_positions", False),
            coefficient_float64=getattr(recipe, "precise_local", False),
            compact_c2=getattr(recipe, "compact_c2", False),
        )
        if numeric_elements:
            numeric_out.append(frame["numeric"])
    meta, origin, spacing, _source = _prepare(
        p,
        value,
        chart,
        recipe,
        moments,
        width_table=routing.get("width_table"),
        meta_out=frame["meta"] if frame is not None else None,
    )
    _stamp("group", 0)
    if getattr(recipe, "fresh_partition", False):
        from .fresh_partition_kernels import partition

        order = (
            frame["order"]
            if frame is not None
            else p.new_empty((a,), dtype=torch.int64)
        )
        # The diagnostic count stays in routing after backward. Keep it outside
        # the frame arena so it cannot retain the entire prior frame allocation.
        counts = p.new_zeros((2,), dtype=torch.int32)
        partition[(tr.cdiv(a, 128),)](
            meta,
            origin,
            spacing,
            order,
            counts,
            a,
            *sizes,
            recipe.sigma_cutoff,
            recipe.reciprocal,
            num_warps=4,
            enable_fp_fusion=False,
        )
        count = counts[:1]
        routing["count"] = count
    else:
        order, count = placement(meta, origin, spacing, sizes, recipe, routing)
    _stamp("group", 1)
    _stamp("pack", 0)
    if not getattr(recipe, "canonical_meta", False):
        meta = meta.index_select(0, order)
    _stamp("pack", 1)
    _stamp("coeff_prepare", 0)
    injection = (
        frame["injection"]
        if frame is not None
        else x.new_empty((2, a), dtype=torch.int16)
        if endpoints
        else x.new_empty(0, dtype=torch.int16)
    )
    pos = (
        frame["pos"] if frame is not None else x.new_empty((2, 6, a), dtype=torch.int32)
    )
    layout = getattr(recipe, "coefficient_layout", "soa")
    co = (
        frame["co"]
        if frame is not None
        else x.new_empty(
            (80 * a,)
            if getattr(recipe, "integer_query", False)
            else (2, 3, 6, a)
            if layout == "soa"
            else (2, a, 6, 4 if layout == "aos4" else 3)
        )
    )
    half = frame["half"] if frame is not None else x.new_empty((2, a))
    bp = (
        frame["bp"]
        if frame is not None
        else x.new_empty((2, 2, a), dtype=torch.int32)
        if shards
        else x.new_empty(0, dtype=torch.int32)
    )
    active = (
        (
            frame["active"].zero_()
            if frame is not None
            else x.new_zeros(2, dtype=torch.int32)
        )
        if fused_levels
        else None
    )
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
        COMPACT_C2=getattr(recipe, "compact_c2", False),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("coeff_prepare", compiled)
    if getattr(recipe, "integer_query", False):
        from . import integer_query_kernels as iq

        compiled = iq.prepare[(tr.cdiv(a, 64),)](
            meta,
            co,
            count,
            _source_order(order, recipe),
            origin,
            spacing,
            a,
            *sizes,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("coeff_prepare_integer", compiled)
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
        ctx.workspace_saved = False
        ctx.empty = not a
        if not a:
            ctx.save_for_backward(x)
            return x.new_zeros((b, sizes[0]))
        shared_numeric = [] if getattr(recipe, "whole_frame", False) else None
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
        ) = _frame(
            x,
            p,
            value,
            chart,
            recipe,
            moments,
            routing,
            sizes,
            numeric_out=shared_numeric,
        )
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
        workspace = None
        numeric = None
        if getattr(recipe, "numeric_arena", False):
            import threading

            from .frame_buffers import numeric_workspace

            workspace, numeric = numeric_workspace(
                x,
                input_points=_points(sizes[1], recipe),
                output_points=_points(sizes[0], recipe),
                blocks=max(_blocks(n, recipe) for n in sizes),
                batch=b,
                shards=shards,
                storage=shared_numeric[0] if shared_numeric else None,
            )
            ctx.workspace_saved = True
            ctx.workspace_stream = torch.cuda.current_stream(x.device).cuda_stream
            ctx.workspace_lock = threading.Lock()
        bank, delta, border = _bank_and_delta(
            x,
            sizes[1],
            sizes[0],
            recipe,
            "bank_input",
            "delta_output",
            active,
            1,
            0,
            ml,
            bank_buffer=numeric["input"] if numeric is not None else None,
            delta_buffers=(numeric["adjoint"], numeric["border"] if shards else None)
            if numeric is not None
            else None,
        )
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
            _h_batch(b, recipe),
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
            **_h_args(groups, recipe, x, (origin, spacing, sizes)),
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
            *((workspace,) if workspace is not None else ()),
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        lock = getattr(ctx, "workspace_lock", None)
        if lock is None:
            return _Local._backward_impl(ctx, dy)
        # Enqueue a whole backward before another host thread borrows scratch
        # on the same CUDA stream. Other streams allocate independent scratch.
        with lock:
            return _Local._backward_impl(ctx, dy)

    @staticmethod
    def _backward_impl(ctx, dy):
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
        groups = (
            ctx.saved_tensors[15:-1] if ctx.workspace_saved else ctx.saved_tensors[15:]
        )
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
        numeric = None
        if ctx.workspace_saved and (
            torch.cuda.current_stream(dy.device).cuda_stream == ctx.workspace_stream
        ):
            from .frame_buffers import numeric_workspace

            _, numeric = numeric_workspace(
                dy,
                input_points=_points(sizes[1], recipe),
                output_points=_points(sizes[0], recipe),
                blocks=max(_blocks(n, recipe) for n in sizes),
                batch=b,
                shards=shards,
                storage=ctx.saved_tensors[-1],
            )
        adjoint_bank, delta, border = _bank_and_delta(
            dy,
            sizes[0],
            sizes[1],
            recipe,
            "bank_adjoint",
            "delta_input",
            active,
            0,
            1,
            ml,
            bank_buffer=numeric["adjoint"] if numeric is not None else None,
            delta_buffers=(numeric["delta"], numeric["border"] if shards else None)
            if numeric is not None
            else None,
        )
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
            _h_batch(b, recipe),
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
            **_h_args(groups, recipe, x, (origin, spacing, sizes)),
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
        buffers["stream_layout"] = (
            {}
            if getattr(recipe, "fresh_partition", False)
            else make_layout(len(p), sizes, recipe, p.device)
        )
        buffers["stream_layout_key"] = key
    if getattr(recipe, "table_decode", False):
        routing = buffers["stream_layout"]
        if "width_table" not in routing:
            from .table_decode_kernels import initialize

            table = p.new_empty((384,), dtype=torch.float64)
            initialize[(1,)](table, num_warps=4, enable_fp_fusion=False)
            routing["width_table"] = table
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
