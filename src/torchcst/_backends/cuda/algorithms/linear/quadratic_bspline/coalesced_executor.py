"""Prepared endpoint reuse, staged H queries, and segmented output updates."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import coalesced_kernels as hk
from . import kernels as k
from .executor import _bank, _prepare, _report, _stamp
from .h_executor import _coefficients, _events
from .stream_executor import _build_w, _restore
from .stream_layout import make_layout, placement


def _local_bank(x, bases, recipe):
    bank = _bank(x, bases)
    return bank.permute(0, 2, 1).contiguous() if recipe.point_major else bank


def _local_restore(delta, y, label, recipe):
    _stamp("layout_" + label, 0)
    if recipe.point_major:
        delta = delta.permute(0, 2, 1).contiguous()
    _stamp("layout_" + label, 1)
    return _restore(delta, y, label)


def _emit(
    values, delta, meta, cuts, co, count, origin, spacing, n, dim, events, recipe
):
    a, b = len(meta), values.shape[1]
    if recipe.site_owner:
        from . import owner_kernels as ok

        ea, eb = recipe.event_atoms, recipe.event_batch
        chunks = tr.cdiv(4 * a, ea)
        partial = torch.empty((3, chunks, b), dtype=torch.float64, device=values.device)
        compiled = ok.emit[(n - 1, tr.cdiv(b, eb))](
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
            ea,
            eb,
            recipe.coefficient_derivatives,
            num_warps=recipe.event_warps,
            enable_fp_fusion=False,
        )
        ok.zero_partial[(chunks, tr.cdiv(b, eb))](
            values,
            partial,
            meta,
            co,
            *events,
            a,
            b,
            n,
            dim,
            ea,
            eb,
            chunks,
            recipe.coefficient_derivatives,
            num_warps=recipe.event_warps,
            enable_fp_fusion=False,
        )
        delta[:, 0, :].copy_(partial.sum(1))
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
            recipe.point_major,
            recipe.share_reads,
            recipe.reduce_writes,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
    _report("y_inject" if dim == 0 else "dx_inject", compiled)


class _CoalescedH(torch.autograd.Function):
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
        _stamp("input_sort", 0)
        if recipe.sort_input:
            keys = torch.empty(a, device=p.device, dtype=torch.int64)
            hk.input_keys[(tr.cdiv(a, 128),)](
                meta,
                count,
                keys,
                origin,
                spacing,
                a,
                sizes[1],
                128,
                recipe.reciprocal,
                num_warps=4,
                enable_fp_fusion=False,
            )
            rank = keys.argsort(stable=True)
            meta = meta.index_select(0, rank)
            order = order.index_select(0, rank)
        _stamp("input_sort", 1)
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
        bank = _local_bank(x, bases[1], recipe)
        _stamp("bank_input", 1)
        delta = torch.zeros(
            (3, sizes[0] + 1, b) if recipe.point_major else (3, b, sizes[0] + 1),
            dtype=torch.float64,
            device=x.device,
        )
        _stamp("stream_forward", 0)
        split = recipe.h_mode != "fused"
        values = x.new_empty((a, b)) if split else x.new_empty(0)

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
                recipe.point_major,
                recipe.share_reads,
                recipe.reduce_writes,
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
        y = _local_restore(delta, y, "output", recipe)
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
        input_bank = (
            _local_bank(x, ctx.bases[1], recipe) if ctx.needs_input_grad[1] else None
        )
        _stamp("bank_recompute", 1)
        _stamp("bank_adjoint", 0)
        adjoint_bank = _local_bank(dy, ctx.bases[0], recipe)
        _stamp("bank_adjoint", 1)
        delta = torch.zeros(
            (3, sizes[1] + 1, b) if recipe.point_major else (3, b, sizes[1] + 1),
            dtype=torch.float64,
            device=x.device,
        )
        _stamp("stream_backward", 0)
        split = recipe.h_mode != "fused"
        values = (
            x.new_empty((a, b)) if split and ctx.needs_input_grad[0] else x.new_empty(0)
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
                recipe.point_major,
                recipe.share_reads,
                recipe.reduce_writes,
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
            dx = _local_restore(delta, dx, "input", recipe)
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
    return _CoalescedH.apply(
        x.contiguous(),
        p,
        value,
        chart,
        recipe,
        bases,
        moments,
        buffers["stream_layout"],
    )
