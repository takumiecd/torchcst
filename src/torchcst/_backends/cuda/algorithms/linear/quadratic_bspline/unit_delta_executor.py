"""Immutable unit-profile snapshots, float-pair banks, and bounded integer deltas."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import unit_delta_gluon as ug
from . import unit_delta_kernels as uk
from .executor import _prepare, _report, _stamp
from .native_runtime import backward as native_backward
from .native_runtime import forward as native_forward


def _empty(name):
    _stamp(name, 0)
    _stamp(name, 1)


def _name(stage, index, count):
    return f"{stage}_chunk{index}" if count > 1 else stage


def _bank_buffer(x, n, rows):
    return x.new_empty((4, n + 1, rows))


def _bank(x, out, label):
    _stamp(label, 0)
    kernel = uk.bank[(out.shape[2],)](
        x, out, x.shape[1], len(x), out.shape[2], num_warps=4, enable_fp_fusion=False
    )
    _report(label, kernel)
    _stamp(label, 1)
    return out


def _restore(delta, compact, bank, y, recipe, label):
    n, rows = y.shape[1], delta.shape[-1]
    _stamp(label, 0)
    kernel = uk.reduce_shards[(tr.cdiv(n * rows, 128),)](
        delta, compact, n, rows, recipe.unit_shards, num_warps=4, enable_fp_fusion=False
    )
    _report(label + "_reduce", kernel)
    kernel = uk.restore[(len(y),)](
        compact,
        bank,
        y,
        n,
        len(y),
        rows,
        bank.shape[1] - 1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report(label + "_prefix", kernel)
    _stamp(label, 1)


class _Unit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, buffers):
        ctx.recipe = recipe
        ctx.sizes = (chart.grid_shape[0][0], chart.grid_shape[1][0])
        ctx.empty = not len(p)
        if ctx.empty:
            ctx.save_for_backward(x)
            return x.new_zeros((len(x), ctx.sizes[0]))
        no, ni = ctx.sizes
        a, b = len(p), len(x)
        tile = min(b, recipe.batch_chunk)
        chunks = tr.cdiv(b, tile)
        ctx.saved_banks = 0
        meta, origin, spacing, _source = _prepare(p, value, chart, recipe, moments)
        for name in ("group", "pack", "active_levels"):
            _empty(name)
        _stamp("coeff_prepare", 0)
        co = x.new_empty((44, a), dtype=torch.int64)
        ids = x.new_empty((a,), dtype=torch.int64)
        count = x.new_zeros((1,), dtype=torch.int32)
        compiled = uk.prepare[(tr.cdiv(a, 64),)](
            meta,
            co,
            ids,
            count,
            origin,
            spacing,
            a,
            no,
            ni,
            recipe.sigma_cutoff,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("coeff_prepare", compiled)
        _stamp("coeff_prepare", 1)
        buffers["unit_native_count"] = count
        y = x.new_zeros((b, no))
        native_forward(x, y, meta, count, origin, spacing, ctx.sizes, recipe, order=ids)
        delta = x.new_empty((2, recipe.unit_shards, no, tile), dtype=torch.int64)
        compact = x.new_empty((2, no, tile), dtype=torch.int64)
        bank = None
        snapshots = []
        for i, start in enumerate(range(0, b, tile)):
            xs, ys = x[start : start + tile], y[start : start + tile]
            if bank is None or (recipe.save_input_bank and ctx.needs_input_grad[1]):
                bank = _bank_buffer(x, ni, tile)
            _bank(xs, bank, _name("bank_input", i, chunks))
            if recipe.save_input_bank and ctx.needs_input_grad[1]:
                snapshots.append(bank)
            label = _name("delta_output", i, chunks)
            _stamp(label, 0)
            delta.zero_()
            _stamp(label, 1)
            label = _name("stream_forward", i, chunks)
            _stamp(label, 0)
            compiled = ug.forward[(tr.cdiv(a, recipe.axis_atoms),)](
                bank,
                meta,
                co,
                spacing,
                delta,
                a,
                len(xs),
                no,
                ni,
                tile,
                recipe.axis_atoms,
                tr.next_power_of_2(len(xs)),
                recipe.unit_shards,
                recipe.lane_atoms,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            _restore(delta, compact, bank, ys, recipe, _name("scan_output", i, chunks))
        ctx.saved_banks = len(snapshots)
        ctx.save_for_backward(x, meta, co, ids, count, origin, spacing, *snapshots)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            (x,) = ctx.saved_tensors
            return (
                torch.zeros_like(x),
                x.new_empty((0, 4)),
                None,
                None,
                None,
                None,
                None,
            )
        x, meta, co, ids, count, origin, spacing = ctx.saved_tensors[:7]
        saved = ctx.saved_tensors[7:]
        recipe = ctx.recipe
        no, ni = ctx.sizes
        a, b = len(meta), len(x)
        tile = min(b, recipe.batch_chunk)
        chunks = tr.cdiv(b, tile)
        need_dx, need_dp = ctx.needs_input_grad[:2]
        dy = dy.contiguous()
        dx, dp = native_backward(
            x,
            dy,
            meta,
            count,
            origin,
            spacing,
            ctx.sizes,
            recipe,
            need_dx,
            need_dp,
            zero_dp=True,
            order=ids,
        )
        input_bank = (
            _bank_buffer(x, ni, tile) if need_dp and not saved else x.new_empty(0)
        )
        adjoint = _bank_buffer(x, no, tile)
        delta = (
            x.new_empty((2, recipe.unit_shards, ni, tile), dtype=torch.int64)
            if need_dx
            else x.new_empty(0, dtype=torch.int64)
        )
        compact = x.new_empty((2, ni, tile), dtype=torch.int64) if need_dx else None
        for i, start in enumerate(range(0, b, tile)):
            xs, dys = x[start : start + tile], dy[start : start + tile]
            label = _name("bank_recompute", i, chunks)
            if need_dp and saved:
                input_bank = saved[i]
                _empty(label)
            elif need_dp:
                _bank(xs, input_bank, label)
            else:
                _empty(label)
            _bank(dys, adjoint, _name("bank_adjoint", i, chunks))
            label = _name("delta_input", i, chunks)
            _stamp(label, 0)
            if need_dx:
                delta.zero_()
            _stamp(label, 1)
            label = _name("stream_backward", i, chunks)
            _stamp(label, 0)
            compiled = ug.backward[(tr.cdiv(a, recipe.axis_atoms),)](
                input_bank,
                adjoint,
                meta,
                co,
                spacing,
                delta,
                dp,
                a,
                len(xs),
                no,
                ni,
                tile,
                recipe.axis_atoms,
                tr.next_power_of_2(len(xs)),
                recipe.unit_shards,
                need_dx,
                need_dp,
                i > 0,
                recipe.contract_float32,
                recipe.lane_atoms,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report(label, compiled)
            _stamp(label, 1)
            if need_dx:
                _restore(
                    delta,
                    compact,
                    adjoint,
                    dx[start : start + tile],
                    recipe,
                    _name("scan_input", i, chunks),
                )
            else:
                _empty(_name("scan_input", i, chunks))
        _empty("unpack")
        return dx, dp if need_dp else None, None, None, None, None, None


def apply(x, p, value, chart, recipe, bases, moments, buffers):
    if len(p) > 60000:
        raise ValueError("two-word profile accumulation supports at most 60000 atoms")
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
    return _Unit.apply(x.contiguous(), p, value, chart, recipe, moments, buffers)
