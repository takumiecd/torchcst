"""Shared immutable coefficients with bounded batch workspaces and full source VJP."""

import torch
from torch.autograd.function import once_differentiable

from .executor import _report, _stamp
from .local_executor import (
    _bank,
    _bank_and_delta,
    _blocks,
    _cache_args,
    _delta_dtype,
    _finish_native,
    _frame,
    _group_tables,
    _h_args,
    _h_batch,
    _h_launch,
    _Local,
    _native_h_args,
    _native_part,
    _points,
    _query_buffer,
    _restore,
    _source_order,
)
from .native_runtime import backward as native_backward
from .native_runtime import forward as native_forward


def _name(stage, index):
    return f"{stage}_chunk{index}"


def _empty(stage):
    _stamp(stage, 0)
    _stamp(stage, 1)


def _workspace(x, n, batch, recipe):
    return x.new_empty((3, _points(n, recipe), batch), dtype=_delta_dtype(x, recipe))


def _border(x, n, batch, recipe):
    return x.new_empty(
        (3, _blocks(n, recipe), recipe.boundary_shards, batch),
        dtype=_delta_dtype(x, recipe),
    )


class _Chunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, routing):
        if not len(p):
            return _Local.forward(ctx, x, p, value, chart, recipe, moments, routing)
        ctx.empty = False
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        ctx.recipe, ctx.sizes = recipe, sizes
        a, b = len(p), len(x)
        tile = recipe.batch_chunk
        retain_bank = (
            getattr(recipe, "save_input_bank", False) and ctx.needs_input_grad[1]
        )
        ctx.saved_bank_count = 0
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
        bank = None if retain_bank else _query_buffer(x, sizes[1], tile, recipe)
        saved_banks = []
        delta = _workspace(x, sizes[0], tile, recipe)
        border = _border(x, sizes[0], tile, recipe)
        empty = x.new_empty(0)
        for index, start in enumerate(range(0, b, tile)):
            xs, ys = x[start : start + tile], y[start : start + tile]
            actual = len(xs)
            if retain_bank:
                # Every chunk owns its forward snapshot. A later chunk/forward
                # must never overwrite bank values needed by retained backward.
                bank = _query_buffer(x, sizes[1], tile, recipe)
            bank, _, _ = _bank_and_delta(
                xs,
                sizes[1],
                sizes[0],
                recipe,
                _name("bank_input", index),
                _name("delta_output", index),
                active,
                1,
                0,
                ml,
                bank,
                (delta, border),
            )
            if retain_bank:
                saved_banks.append(bank)
            label = _name("stream_forward", index)
            _stamp(label, 0)
            compiled = _h_launch("forward", a, recipe)(
                bank,
                meta,
                pos,
                co,
                half,
                empty,
                empty,
                delta,
                count,
                a,
                actual,
                _points(sizes[1], recipe),
                _points(sizes[0], recipe),
                recipe.axis_atoms,
                _h_batch(actual, recipe),
                "none",
                border,
                bp,
                shards,
                _blocks(sizes[0], recipe),
                injection,
                endpoints,
                recipe.shard_origin,
                tile - actual,
                Order=_source_order(order, recipe),
                LAYOUT=getattr(recipe, "coefficient_layout", "soa"),
                **_h_args(groups, recipe, x, (origin, spacing, sizes)),
                **_native_h_args(
                    xs,
                    None,
                    _native_part(native_y, start, actual, recipe),
                    sizes,
                    recipe,
                ),
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            _finish_native(
                ys,
                _native_part(native_y, start, actual, recipe),
                _name("native_finish_forward", index),
                recipe,
            )
            _restore(
                delta,
                border,
                ys,
                sizes[0],
                recipe,
                _name("scan_output", index),
                active,
                0,
                ml,
            )
        ctx.saved_bank_count = len(saved_banks)
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
            active,
            bp,
            injection,
            *groups,
            *saved_banks,
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            return _Local.backward(ctx, dy)
        meta, x, origin, spacing, order, count, pos, co, half, active, bp, injection = (
            ctx.saved_tensors[:12]
        )
        saved_count = ctx.saved_bank_count
        groups = (
            ctx.saved_tensors[12:-saved_count]
            if saved_count
            else ctx.saved_tensors[12:]
        )
        saved_banks = ctx.saved_tensors[-saved_count:] if saved_count else ()
        dy = dy.contiguous()
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b = len(meta), len(x)
        tile = recipe.batch_chunk
        ml = max(sizes).bit_length() - recipe.min_block_log
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
            zero_dp=True,
            order=_source_order(order, recipe),
            cache=_cache_args(pos, co, active, recipe),
        )
        if getattr(recipe, "unified_native", False):
            dx, dp, native_dx = native_result
        else:
            dx, dp = native_result
            native_dx = None
        input_bank = (
            _query_buffer(x, sizes[1], tile, recipe)
            if need_dp and not saved_count
            else None
        )
        adjoint_bank = _query_buffer(x, sizes[0], tile, recipe)
        delta = _workspace(x, sizes[1], tile, recipe)
        border = _border(x, sizes[1], tile, recipe)
        empty = x.new_empty(0)
        for index, start in enumerate(range(0, b, tile)):
            xs, dys = x[start : start + tile], dy[start : start + tile]
            actual = len(xs)
            if need_dp and saved_count:
                input_bank = saved_banks[index]
                _empty(_name("bank_recompute", index))
            elif need_dp:
                _bank(
                    xs,
                    sizes[1],
                    recipe,
                    _name("bank_recompute", index),
                    active,
                    1,
                    ml,
                    input_bank,
                )
            else:
                _empty(_name("bank_recompute", index))
            adjoint_bank, _, _ = _bank_and_delta(
                dys,
                sizes[0],
                sizes[1],
                recipe,
                _name("bank_adjoint", index),
                _name("delta_input", index),
                active,
                0,
                1,
                ml,
                adjoint_bank,
                (delta, border),
            )
            label = _name("stream_backward", index)
            _stamp(label, 0)
            compiled = _h_launch("backward", a, recipe)(
                input_bank,
                adjoint_bank,
                meta,
                pos,
                co,
                half,
                empty,
                empty,
                delta,
                dp,
                count,
                a,
                actual,
                _points(sizes[1], recipe),
                _points(sizes[0], recipe),
                recipe.axis_atoms,
                _h_batch(actual, recipe),
                "none",
                need_dx,
                need_dp,
                recipe.contract_float32,
                border,
                bp,
                recipe.boundary_shards,
                _blocks(sizes[1], recipe),
                injection,
                recipe.skip_endpoints,
                recipe.shard_origin,
                tile - actual,
                True,
                Order=_source_order(order, recipe),
                LAYOUT=getattr(recipe, "coefficient_layout", "soa"),
                **_h_args(groups, recipe, x, (origin, spacing, sizes)),
                **_native_h_args(
                    xs,
                    dys,
                    _native_part(native_dx, start, actual, recipe),
                    sizes,
                    recipe,
                ),
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            if need_dx:
                _finish_native(
                    dx[start : start + tile],
                    _native_part(native_dx, start, actual, recipe),
                    _name("native_finish_backward", index),
                    recipe,
                )
                _restore(
                    delta,
                    border,
                    dx[start : start + tile],
                    sizes[1],
                    recipe,
                    _name("scan_input", index),
                    active,
                    1,
                    ml,
                )
            else:
                _empty(_name("scan_input", index))
        _stamp("unpack", 0)
        if need_dp and not getattr(recipe, "canonical_meta", False):
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return dx, dp if need_dp else None, None, None, None, None, None
