"""Shared event ownership with a bounded weighted-H/G workspace."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import local_owner_kernels as ok
from . import owner_runtime as owner
from .chunk_executor import _empty, _name, _workspace
from .executor import _report, _stamp
from .local_executor import _bank, _frame, _Local, _points, _restore
from .native_runtime import backward as native_backward
from .native_runtime import forward as native_forward


def _view(arena, points, tile):
    return arena.view(-1)[: 3 * points * tile].view(3, points, tile)


def _zero(delta, label):
    _stamp(label, 0)
    delta.zero_()
    _stamp(label, 1)


class _Owner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, routing):
        if not len(p):
            return _Local.forward(ctx, x, p, value, chart, recipe, moments, routing)
        ctx.empty = False
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        ctx.recipe, ctx.sizes = recipe, sizes
        a, b, tile = len(p), len(x), recipe.batch_chunk
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
            _,
            _,
        ) = _frame(
            x, p, value, chart, recipe, moments, routing, sizes, build_border=False
        )
        tables = owner.prepare(x, pos, injection, count, a, sizes, recipe)
        po, pi = (_points(n, recipe) for n in sizes)
        arena = x.new_empty((3, max(pi, po), tile))
        bank, delta = _view(arena, pi, tile), _view(arena, po, tile)
        weight = x.new_empty((a, tile))
        y = x.new_zeros((b, sizes[0]))
        native_forward(x, y, meta, count, origin, spacing, sizes, recipe)
        for index, start in enumerate(range(0, b, tile)):
            xs, ys = x[start : start + tile], y[start : start + tile]
            actual = len(xs)
            _bank(xs, sizes[1], recipe, _name("bank_input", index), active, 1, ml, bank)
            label = _name("stream_forward", index)
            _stamp(label, 0)
            compiled = ok.project_forward[(tr.cdiv(a, recipe.axis_atoms),)](
                bank,
                meta,
                pos,
                co,
                half,
                count,
                weight,
                a,
                actual,
                pi,
                tile,
                recipe.axis_atoms,
                tr.next_power_of_2(actual),
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            # Projection has finished reading the bank before it becomes delta.
            _zero(delta, _name("delta_output", index))
            owner.inject(
                weight,
                co,
                tables[0],
                delta,
                actual,
                a,
                0,
                recipe,
                _name("owner_output", index),
            )
            _restore(
                delta,
                None,
                ys,
                sizes[0],
                recipe,
                _name("scan_output", index),
                active,
                0,
                ml,
            )
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
            *(v for table in tables for v in table),
        )
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            return _Local.backward(ctx, dy)
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
            active,
            _bp,
            _injection,
            *raw,
        ) = ctx.saved_tensors
        tables = [raw[:5], raw[5:]]
        dy = dy.contiguous()
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b, tile = len(meta), len(x), recipe.batch_chunk
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
        pi, po = _points(sizes[1], recipe), _points(sizes[0], recipe)
        arena = _workspace(x, sizes[1], tile, recipe)
        input_bank = arena if need_dp else None
        delta = arena
        adjoint_bank = _workspace(x, sizes[0], tile, recipe)
        weight = x.new_empty((a, tile)) if need_dx else x.new_empty(0)
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
            label = _name("stream_backward", index)
            _stamp(label, 0)
            compiled = ok.project_backward[(tr.cdiv(a, recipe.axis_atoms),)](
                input_bank,
                adjoint_bank,
                meta,
                pos,
                co,
                half,
                count,
                weight,
                dp,
                a,
                actual,
                pi,
                po,
                tile,
                recipe.axis_atoms,
                tr.next_power_of_2(actual),
                need_dx,
                need_dp,
                recipe.contract_float32,
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            if need_dx:
                _zero(delta, _name("delta_input", index))
                owner.inject(
                    weight,
                    co,
                    tables[1],
                    delta,
                    actual,
                    a,
                    1,
                    recipe,
                    _name("owner_input", index),
                )
                _restore(
                    delta,
                    None,
                    dx[start : start + tile],
                    sizes[1],
                    recipe,
                    _name("scan_input", index),
                    active,
                    1,
                    ml,
                )
            else:
                for name in ("delta_input", "owner_input", "scan_input"):
                    _empty(_name(name, index))
        _stamp("unpack", 0)
        if need_dp:
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return dx, dp if need_dp else None, None, None, None, None, None
