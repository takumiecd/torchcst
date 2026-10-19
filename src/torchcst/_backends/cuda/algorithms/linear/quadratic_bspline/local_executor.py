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


def _bank(x, n, recipe, label, active, dim, ml):
    bank = x.new_empty(
        (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0))
    )
    _stamp(label, 0)
    for k, sizes in enumerate(_groups(n, recipe)):
        compiled = lk.bank[
            (sum(n // s for s in sizes), tr.cdiv(len(x), recipe.bank_batch))
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
            getattr(recipe, "bank_padding", 0),
            getattr(recipe, "min_block_log", 3),
            num_warps=recipe.bank_warps,
            enable_fp_fusion=False,
        )
        _report(label + f"_{k}", compiled)
    _stamp(label, 1)
    return bank


def _restore(delta, border, y, n, recipe, label, active, dim, ml):
    _stamp(label, 0)
    for k, sizes in enumerate(_groups(n, recipe)):
        compiled = lk.restore[
            (sum(n // s for s in sizes), tr.cdiv(len(y), recipe.bank_batch))
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
            getattr(recipe, "boundary_shards", 0),
            getattr(recipe, "bank_padding", 0),
            getattr(recipe, "min_block_log", 3),
            num_warps=recipe.bank_warps,
            enable_fp_fusion=False,
        )
        _report(label + f"_{k}", compiled)
    _stamp(label, 1)
    return y


def _delta(x, n, recipe, active, dim, ml):
    label = "delta_output" if dim == 0 else "delta_input"
    _stamp(label, 0)
    if getattr(recipe, "active_levels", False):
        delta = x.new_empty(
            (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0))
        )
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
                getattr(recipe, "bank_padding", 0),
                getattr(recipe, "min_block_log", 3),
                num_warps=recipe.bank_warps,
                enable_fp_fusion=False,
            )
            _report(label + f"_{k}", compiled)
    else:
        delta = x.new_zeros(
            (3, _points(n, recipe), len(x) + getattr(recipe, "bank_padding", 0))
        )
    shards = getattr(recipe, "boundary_shards", 0)
    border = (
        x.new_zeros(
            (3, _blocks(n, recipe), shards, len(x) + getattr(recipe, "bank_padding", 0))
        )
        if shards
        else None
    )
    _stamp(label, 1)
    return delta, border


class _Local(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, routing):
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        a, b = len(p), len(x)
        ctx.recipe, ctx.sizes = recipe, sizes
        ctx.mode = recipe.save_h if ctx.needs_input_grad[1] else "none"
        ctx.empty = not a
        if not a:
            ctx.save_for_backward(x)
            return x.new_zeros((b, sizes[0]))
        meta, origin, spacing, _source = _prepare(p, value, chart, recipe, moments)
        _stamp("group", 0)
        order, count = placement(meta, origin, spacing, sizes, recipe, routing)
        _stamp("group", 1)
        _stamp("pack", 0)
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
        co = x.new_empty((2, 3, 6, a))
        half = x.new_empty((2, a))
        shards = getattr(recipe, "boundary_shards", 0)
        bp = (
            x.new_empty((2, 2, a), dtype=torch.int32)
            if shards
            else x.new_empty(0, dtype=torch.int32)
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
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("coeff_prepare", compiled)
        _stamp("coeff_prepare", 1)
        ml = max(sizes).bit_length() - getattr(recipe, "min_block_log", 3)
        _stamp("active_levels", 0)
        if getattr(recipe, "active_levels", False):
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
        else:
            active = x.new_empty(0, dtype=torch.int32)
        _stamp("active_levels", 1)
        y = x.new_zeros((b, sizes[0]))
        native_forward(x, y, meta, count, origin, spacing, sizes, recipe)
        bank = _bank(x, sizes[1], recipe, "bank_input", active, 1, ml)
        delta, border = _delta(x, sizes[0], recipe, active, 0, ml)
        h = x.new_empty((a, b)) if ctx.mode != "none" else x.new_empty(0)
        hp = x.new_empty((a, b)) if ctx.mode == "hp" else x.new_empty(0)
        _stamp("stream_forward", 0)
        compiled = lk.forward[(tr.cdiv(a, recipe.axis_atoms),)](
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
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _report("stream_forward", compiled)
        _stamp("stream_forward", 1)
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
        ) = ctx.saved_tensors
        dy = dy.contiguous()
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b = len(meta), len(x)
        ml = max(sizes).bit_length() - getattr(recipe, "min_block_log", 3)
        shards = getattr(recipe, "boundary_shards", 0)
        endpoints = getattr(recipe, "skip_endpoints", False)
        need_dx, need_dp = ctx.needs_input_grad[:2]
        dx, dp = native_backward(
            x, dy, meta, count, origin, spacing, sizes, recipe, need_dx, need_dp
        )
        if need_dp and ctx.mode != "hp":
            input_bank = _bank(x, sizes[1], recipe, "bank_recompute", active, 1, ml)
        else:
            input_bank = None
            _stamp("bank_recompute", 0)
            _stamp("bank_recompute", 1)
        adjoint_bank = _bank(dy, sizes[0], recipe, "bank_adjoint", active, 0, ml)
        delta, border = _delta(x, sizes[1], recipe, active, 1, ml)
        _stamp("stream_backward", 0)
        compiled = lk.backward[(tr.cdiv(a, recipe.axis_atoms),)](
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
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _report("stream_backward", compiled)
        _stamp("stream_backward", 1)
        if need_dx:
            dx = _restore(
                delta, border, dx, sizes[1], recipe, "scan_input", active, 1, ml
            )
        else:
            _stamp("scan_input", 0)
            _stamp("scan_input", 1)
        _stamp("unpack", 0)
        if need_dp:
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
    return _Local.apply(
        x.contiguous(), p, value, chart, recipe, moments, buffers["stream_layout"]
    )
