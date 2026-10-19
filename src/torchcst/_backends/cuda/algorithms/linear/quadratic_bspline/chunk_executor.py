"""Shared immutable coefficients with bounded batch workspaces and full source VJP."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import local_kernels as lk
from .executor import _report, _stamp
from .local_executor import _bank, _blocks, _delta, _frame, _Local, _points, _restore
from .native_runtime import backward as native_backward
from .native_runtime import forward as native_forward


def _name(stage, index):
    return f"{stage}_chunk{index}"


def _empty(stage):
    _stamp(stage, 0)
    _stamp(stage, 1)


def _workspace(x, n, batch, recipe):
    return x.new_empty((3, _points(n, recipe), batch))


def _border(x, n, batch, recipe):
    return x.new_empty((3, _blocks(n, recipe), recipe.boundary_shards, batch))


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
        y = x.new_zeros((b, sizes[0]))
        native_forward(x, y, meta, count, origin, spacing, sizes, recipe)
        bank = _workspace(x, sizes[1], tile, recipe)
        delta = _workspace(x, sizes[0], tile, recipe)
        border = _border(x, sizes[0], tile, recipe)
        empty = x.new_empty(0)
        for index, start in enumerate(range(0, b, tile)):
            xs, ys = x[start : start + tile], y[start : start + tile]
            actual = len(xs)
            _bank(xs, sizes[1], recipe, _name("bank_input", index), active, 1, ml, bank)
            _delta(
                xs,
                sizes[0],
                recipe,
                active,
                0,
                ml,
                (delta, border),
                _name("delta_output", index),
            )
            label = _name("stream_forward", index)
            _stamp(label, 0)
            compiled = lk.forward[(tr.cdiv(a, recipe.axis_atoms),)](
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
                tr.next_power_of_2(actual),
                "none",
                border,
                bp,
                shards,
                _blocks(sizes[0], recipe),
                injection,
                endpoints,
                recipe.shard_origin,
                tile - actual,
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
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
        ctx.save_for_backward(
            meta, x, origin, spacing, order, count, pos, co, half, active, bp, injection
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            return _Local.backward(ctx, dy)
        meta, x, origin, spacing, order, count, pos, co, half, active, bp, injection = (
            ctx.saved_tensors
        )
        dy = dy.contiguous()
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b = len(meta), len(x)
        tile = recipe.batch_chunk
        ml = max(sizes).bit_length() - recipe.min_block_log
        need_dx, need_dp = ctx.needs_input_grad[:2]
        dx, dp = native_backward(
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
        )
        input_bank = _workspace(x, sizes[1], tile, recipe) if need_dp else None
        adjoint_bank = _workspace(x, sizes[0], tile, recipe)
        delta = _workspace(x, sizes[1], tile, recipe)
        border = _border(x, sizes[1], tile, recipe)
        empty = x.new_empty(0)
        for index, start in enumerate(range(0, b, tile)):
            xs, dys = x[start : start + tile], dy[start : start + tile]
            actual = len(xs)
            if need_dp:
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
            _bank(
                dys,
                sizes[0],
                recipe,
                _name("bank_adjoint", index),
                active,
                0,
                ml,
                adjoint_bank,
            )
            _delta(
                xs,
                sizes[1],
                recipe,
                active,
                1,
                ml,
                (delta, border),
                _name("delta_input", index),
            )
            label = _name("stream_backward", index)
            _stamp(label, 0)
            compiled = lk.backward[(tr.cdiv(a, recipe.axis_atoms),)](
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
                tr.next_power_of_2(actual),
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
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            if need_dx:
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
        if need_dp:
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return dx, dp if need_dp else None, None, None, None, None, None
