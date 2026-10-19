"""Immutable per-forward H snapshots and staged/fused backward comparisons."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import coalesced_kernels as ck
from . import kernels as k
from . import saved_kernels as hk
from .coalesced_executor import _emit, _local_bank, _local_restore
from .executor import _prepare, _report, _stamp
from .h_executor import _coefficients, _events
from .stream_executor import _build_w
from .stream_layout import make_layout, placement


class _SavedH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, bases, moments, routing):
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        a, b = len(p), len(x)
        ctx.sizes, ctx.recipe, ctx.bases = sizes, recipe, bases
        ctx.saved_mode = recipe.save_h if ctx.needs_input_grad[1] else "none"
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
        _stamp("input_sort", 1)
        cuts, co = _coefficients(meta, origin, spacing, count, sizes, recipe)
        if getattr(recipe, "adjoint_h", False):
            from .adjoint_runtime import order as adjoint_order

            backward_order = adjoint_order(cuts, count, sizes, recipe)
        else:
            backward_order = x.new_empty(0, dtype=torch.int64)
        _out_order, _out_ends = _events(cuts, count, a, sizes[0], 0, recipe)
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
        mode = ctx.saved_mode
        values = x.new_empty((a, b)) if mode != "none" else x.new_empty(0)
        saved_hp = x.new_empty((a, b)) if mode == "hp" else x.new_empty(0)
        _stamp("h_query", 0)
        _stamp("h_query", 1)
        _stamp("y_inject", 0)
        _stamp("y_inject", 1)
        kernel = ck.forward if mode == "none" else hk.forward
        prefix = (
            (bank, meta, cuts, co, values)
            if mode == "none"
            else (bank, meta, cuts, co, values, saved_hp)
        )
        mode_arg = False if mode == "none" else mode
        compiled = kernel[(tr.cdiv(a, recipe.axis_atoms),)](
            *prefix,
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
            mode_arg,
            recipe.edge_reduce,
            recipe.point_major,
            recipe.share_reads,
            recipe.reduce_writes,
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _report("stream_forward", compiled)
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
            values,
            saved_hp,
            backward_order,
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
            saved_h,
            saved_hp,
            backward_order,
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
            _local_bank(x, ctx.bases[1], recipe)
            if ctx.needs_input_grad[1] and ctx.saved_mode != "hp"
            else None
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
        mode = ctx.saved_mode
        vjp, need_dx = ctx.needs_input_grad[1], ctx.needs_input_grad[0]
        if not vjp:
            dp = dy.new_empty(0)
        _stamp("back_query", 0)
        _stamp("back_query", 1)
        if getattr(recipe, "adjoint_h", False):
            from .adjoint_runtime import backward as adjoint_backward

            for stage in ("h_recompute", "g_recompute", "atom_contract", "dx_inject"):
                _stamp(stage, 0)
                _stamp(stage, 1)
            delta = adjoint_backward(
                backward_order,
                input_bank,
                saved_h,
                saved_hp,
                adjoint_bank,
                meta,
                cuts,
                co,
                delta,
                dp,
                count,
                origin,
                spacing,
                sizes,
                b,
                recipe,
                mode,
                need_dx,
                vjp,
            )
        elif recipe.backward_mode == "staged":
            h = (
                saved_h
                if mode != "none"
                else (x.new_empty((a, b)) if vjp else x.new_empty(0))
            )
            hp = (
                saved_hp
                if mode == "hp"
                else (x.new_empty((a, b)) if vjp else x.new_empty(0))
            )
            g = x.new_empty((a, b))
            gp = x.new_empty((a, b)) if vjp else x.new_empty(0)
            _stamp("h_recompute", 0)
            if vjp and mode != "hp":
                compiled = hk.input_query[(tr.cdiv(a, recipe.axis_atoms),)](
                    input_bank,
                    saved_h,
                    h,
                    hp,
                    meta,
                    cuts,
                    co,
                    count,
                    origin,
                    spacing,
                    a,
                    b,
                    sizes[1],
                    recipe.axis_atoms,
                    tr.next_power_of_2(b),
                    recipe.reciprocal,
                    recipe.coefficient_derivatives,
                    mode,
                    num_warps=recipe.axis_warps,
                    enable_fp_fusion=False,
                )
                _report("h_recompute", compiled)
            _stamp("h_recompute", 1)
            _stamp("g_recompute", 0)
            compiled = hk.adjoint_query[(tr.cdiv(a, recipe.axis_atoms),)](
                adjoint_bank,
                g,
                gp,
                meta,
                cuts,
                co,
                count,
                origin,
                spacing,
                a,
                b,
                sizes[0],
                recipe.axis_atoms,
                tr.next_power_of_2(b),
                recipe.reciprocal,
                recipe.coefficient_derivatives,
                vjp,
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report("g_recompute", compiled)
            _stamp("g_recompute", 1)
            _stamp("atom_contract", 0)
            if vjp:
                compiled = hk.contract[(tr.cdiv(a, recipe.axis_atoms),)](
                    h,
                    hp,
                    g,
                    gp,
                    meta,
                    dp,
                    count,
                    a,
                    b,
                    recipe.axis_atoms,
                    tr.next_power_of_2(b),
                    num_warps=recipe.axis_warps,
                    enable_fp_fusion=False,
                )
                _report("atom_contract", compiled)
            _stamp("atom_contract", 1)
            _stamp("dx_inject", 0)
            if need_dx:
                _emit(
                    g,
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
        else:
            for stage in ("h_recompute", "g_recompute", "atom_contract", "dx_inject"):
                _stamp(stage, 0)
                _stamp(stage, 1)
            kernel = ck.backward if mode == "none" else hk.backward
            prefix = (
                (input_bank, adjoint_bank)
                if mode == "none"
                else (input_bank, saved_h, saved_hp, adjoint_bank)
            )
            extra = () if mode == "none" else (mode,)
            compiled = kernel[(tr.cdiv(a, recipe.axis_atoms),)](
                *prefix,
                meta,
                cuts,
                co,
                x.new_empty(0),
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
                False,
                recipe.edge_reduce,
                recipe.point_major,
                recipe.share_reads,
                recipe.reduce_writes,
                need_dx,
                vjp,
                *extra,
                num_warps=recipe.axis_warps,
                enable_fp_fusion=False,
            )
            _report("stream_backward", compiled)
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
    return _SavedH.apply(
        x.contiguous(),
        p,
        value,
        chart,
        recipe,
        bases,
        moments,
        buffers["stream_layout"],
    )
