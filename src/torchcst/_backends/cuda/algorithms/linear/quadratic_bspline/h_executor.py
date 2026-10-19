"""Prepared endpoint reuse, staged H queries, and segmented output updates."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import h_kernels as hk
from . import kernels as k
from .executor import _bank, _prepare, _report, _stamp
from .stream_executor import _build_w, _restore
from .stream_layout import make_layout, placement


def _coefficients(meta, origin, spacing, count, sizes, recipe):
    _stamp("coeff_prepare", 0)
    if recipe.coefficient_cache:
        a = len(meta)
        cuts = torch.empty((2, 4, a), device=meta.device, dtype=torch.int32)
        co = torch.empty(
            (2, 5 if recipe.coefficient_derivatives else 3, 4, a),
            device=meta.device,
            dtype=torch.float64,
        )
        compiled = hk.prepare[(tr.cdiv(a, 128),)](
            meta,
            cuts,
            co,
            count,
            origin,
            spacing,
            a,
            *sizes,
            128,
            recipe.reciprocal,
            recipe.coefficient_derivatives,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _report("coeff_prepare", compiled)
    else:
        cuts = torch.empty(0, device=meta.device, dtype=torch.int32)
        co = torch.empty(0, device=meta.device, dtype=torch.float64)
    _stamp("coeff_prepare", 1)
    return cuts, co


def _events(cuts, count, a, n, dim, recipe):
    stage = "event_output" if dim == 0 else "event_input"
    _stamp(stage, 0)
    if recipe.h_mode == "segmented" or getattr(recipe, "site_owner", False):
        hist = torch.zeros(n, device=cuts.device, dtype=torch.int32)
        cursor = torch.zeros_like(hist)
        order = torch.empty(4 * a, device=cuts.device, dtype=torch.int32)
        hk.event_hist[(tr.cdiv(4 * a, 256),)](cuts, count, hist, a, n, dim, 256)
        ends = hist.cumsum(0, dtype=torch.int32)
        hk.event_place[(tr.cdiv(4 * a, 256),)](
            cuts, count, hist, ends, cursor, order, a, n, dim, 256
        )
    else:
        order = ends = torch.empty(0, device=cuts.device, dtype=torch.int32)
    _stamp(stage, 1)
    return order, ends


def _emit(
    values, delta, meta, cuts, co, count, origin, spacing, n, dim, events, recipe
):
    a, b = len(meta), len(values)
    if recipe.h_mode == "segmented":
        compiled = hk.event_inject[
            (tr.cdiv(4 * a, recipe.event_atoms), tr.cdiv(b, recipe.event_batch))
        ](
            values,
            delta,
            meta,
            cuts,
            co,
            *events,
            a,
            b,
            n,
            dim,
            recipe.event_atoms,
            recipe.event_batch,
            recipe.coefficient_derivatives,
            num_warps=recipe.event_warps,
            enable_fp_fusion=False,
        )
    else:
        compiled = hk.inject[(tr.cdiv(a, recipe.axis_atoms),)](
            values,
            delta,
            meta,
            cuts,
            co,
            count,
            origin,
            spacing,
            a,
            b,
            n,
            dim,
            recipe.axis_atoms,
            tr.next_power_of_2(b),
            recipe.reciprocal,
            recipe.coefficient_cache,
            recipe.coefficient_derivatives,
            recipe.edge_reduce,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    _report("y_inject" if dim == 0 else "dx_inject", compiled)


class _PreparedH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, bases, moments, routing):
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        a, b = len(p), len(x)
        ctx.sizes, ctx.recipe, ctx.bases = sizes, recipe, bases
        ctx.empty = not a
        if not a:
            ctx.save_for_backward(x)
            return x.new_zeros((b, sizes[0]))
        meta, origin, spacing, _source = _prepare(p, value, chart, recipe, moments)
        _stamp("group", 0)
        order, count = placement(meta, origin, spacing, sizes, recipe, routing)
        _stamp("group", 1)
        _stamp("pack", 0)
        # Physical execution rows are contiguous; source IDs and Adam state stay canonical.
        meta = meta.index_select(0, order)
        _stamp("pack", 1)
        cuts, co = _coefficients(meta, origin, spacing, count, sizes, recipe)
        out_order, out_ends = _events(cuts, count, a, sizes[0], 0, recipe)
        in_order, in_ends = _events(cuts, count, a, sizes[1], 1, recipe)
        w = _build_w(meta, origin, spacing, count, sizes, recipe, x.device, "w_build")
        _stamp("y_gemm", 0)
        y = x @ w.T
        _stamp("y_gemm", 1)
        saved_w = w if recipe.save_w else x.new_empty(0)
        del w
        _stamp("bank_input", 0)
        bank = _bank(x, bases[1])
        _stamp("bank_input", 1)
        delta = torch.zeros((3, b, sizes[0] + 1), dtype=torch.float64, device=x.device)
        _stamp("stream_forward", 0)
        split = recipe.h_mode != "fused"
        values = x.new_empty((b, a)) if split else x.new_empty(0)

        def launch():
            compiled = hk.forward[(tr.cdiv(a, recipe.axis_atoms),)](
                bank,
                meta,
                cuts,
                co,
                values,
                delta,
                count,
                origin,
                spacing,
                a,
                b,
                *sizes,
                recipe.axis_atoms,
                tr.next_power_of_2(b),
                recipe.reciprocal,
                recipe.coefficient_cache,
                recipe.coefficient_derivatives,
                split,
                recipe.edge_reduce,
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report("stream_forward", compiled)

        _stamp("h_query", 0)
        if split:
            launch()
        _stamp("h_query", 1)
        _stamp("y_inject", 0)
        if split:
            _emit(
                values,
                delta,
                meta,
                cuts,
                co,
                count,
                origin,
                spacing,
                sizes[0],
                0,
                (out_order, out_ends),
                recipe,
            )
        _stamp("y_inject", 1)
        if not split:
            launch()
        _stamp("stream_forward", 1)
        y = _restore(delta, y, "output")
        ctx.save_for_backward(
            meta,
            x,
            saved_w,
            origin,
            spacing,
            order,
            count,
            cuts,
            co,
            in_order,
            in_ends,
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
                None,
            )
        (
            meta,
            x,
            w,
            origin,
            spacing,
            order,
            count,
            cuts,
            co,
            in_order,
            in_ends,
        ) = ctx.saved_tensors
        recipe, sizes = ctx.recipe, ctx.sizes
        a, b = len(meta), len(dy)
        dy = dy.contiguous()
        dx = dp = None
        if not recipe.save_w and ctx.needs_input_grad[0]:
            w = _build_w(
                meta, origin, spacing, count, sizes, recipe, x.device, "w_rebuild"
            )
        else:
            _stamp("w_rebuild", 0)
            _stamp("w_rebuild", 1)
        _stamp("dx_gemm", 0)
        if ctx.needs_input_grad[0]:
            dx = dy @ w
        _stamp("dx_gemm", 1)
        _stamp("dw_gemm", 0)
        if ctx.needs_input_grad[1]:
            if not recipe.save_w and ctx.needs_input_grad[0]:
                dw = torch.mm(dy.T, x, out=w)
            else:
                dw = dy.T @ x
        _stamp("dw_gemm", 1)
        _stamp("matrix_atom_vjp", 0)
        if ctx.needs_input_grad[1]:
            dp = dy.new_empty((a, 4))
            compiled = k.matrix_vjp[(tr.cdiv(a, recipe.matrix_atoms),)](
                meta,
                dw,
                dp,
                origin,
                spacing,
                a,
                *sizes,
                recipe.matrix_sites,
                recipe.matrix_atoms,
                recipe.reciprocal,
                True,
                Count=count,
                PARTITIONED=True,
                num_warps=recipe.matrix_warps,
                enable_fp_fusion=False,
            )
            _report("matrix_atom_vjp", compiled)
            del dw
        _stamp("matrix_atom_vjp", 1)
        del w
        _stamp("bank_recompute", 0)
        input_bank = _bank(x, ctx.bases[1]) if ctx.needs_input_grad[1] else None
        _stamp("bank_recompute", 1)
        _stamp("bank_adjoint", 0)
        adjoint_bank = _bank(dy, ctx.bases[0])
        _stamp("bank_adjoint", 1)
        delta = torch.zeros((3, b, sizes[1] + 1), dtype=torch.float64, device=x.device)
        _stamp("stream_backward", 0)
        split = recipe.h_mode != "fused"
        values = (
            x.new_empty((b, a)) if split and ctx.needs_input_grad[0] else x.new_empty(0)
        )
        if not ctx.needs_input_grad[1]:
            dp = dy.new_empty(0)

        def launch():
            compiled = hk.backward[(tr.cdiv(a, recipe.axis_atoms),)](
                input_bank,
                adjoint_bank,
                meta,
                cuts,
                co,
                values,
                delta,
                dp,
                count,
                origin,
                spacing,
                a,
                b,
                *sizes,
                recipe.axis_atoms,
                tr.next_power_of_2(b),
                recipe.reciprocal,
                recipe.coefficient_cache,
                recipe.coefficient_derivatives,
                split,
                recipe.edge_reduce,
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report("stream_backward", compiled)

        _stamp("back_query", 0)
        if split:
            launch()
        _stamp("back_query", 1)
        _stamp("dx_inject", 0)
        if split and ctx.needs_input_grad[0]:
            _emit(
                values,
                delta,
                meta,
                cuts,
                co,
                count,
                origin,
                spacing,
                sizes[1],
                1,
                (in_order, in_ends),
                recipe,
            )
        _stamp("dx_inject", 1)
        if not split:
            launch()
        _stamp("stream_backward", 1)
        if ctx.needs_input_grad[0]:
            dx = _restore(delta, dx, "input")
        _stamp("unpack", 0)
        if ctx.needs_input_grad[1]:
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return (
            dx,
            dp if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def apply(x, p, value, chart, recipe, bases, moments, buffers):
    key = (p.device, len(p), chart.grid_shape, recipe)
    if buffers.get("stream_layout_key") != key:
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        buffers["stream_layout"] = make_layout(len(p), sizes, recipe, p.device)
        buffers["stream_layout_key"] = key
    return _PreparedH.apply(
        x.contiguous(),
        p,
        value,
        chart,
        recipe,
        bases,
        moments,
        buffers["stream_layout"],
    )
