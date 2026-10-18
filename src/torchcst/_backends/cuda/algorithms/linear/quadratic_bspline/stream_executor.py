"""Live exact normalization, persistent placement, and immediate H consumption."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import kernels as k
from . import stream_kernels as sk
from .executor import _bank, _prepare, _report, _stamp
from .stream_layout import make_layout, placement


def _build_w(meta, origin, spacing, count, sizes, recipe, device, stage):
    w = torch.zeros(sizes, dtype=torch.float32, device=device)
    _stamp(stage, 0)
    compiled = k.matrix_build[(tr.cdiv(len(meta), recipe.matrix_atoms),)](
        meta,
        w,
        origin,
        spacing,
        len(meta),
        *sizes,
        recipe.matrix_sites,
        recipe.matrix_atoms,
        recipe.reciprocal,
        Count=count,
        PARTITIONED=True,
        num_warps=recipe.matrix_warps,
        enable_fp_fusion=False,
    )
    _stamp(stage, 1)
    _report(stage, compiled)
    return w


def _restore(delta, y, side):
    b, n = y.shape
    _stamp("scan_" + side, 0)
    scan = delta.cumsum(-1)
    _stamp("scan_" + side, 1)
    _stamp("restore_" + side, 0)
    compiled = k.restore[(tr.cdiv(b * n, 128),)](
        scan, y, n, b, num_warps=4, enable_fp_fusion=False
    )
    _stamp("restore_" + side, 1)
    _report("restore_" + side, compiled)
    return y


class _Streaming(torch.autograd.Function):
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
        compiled = sk.forward[(tr.cdiv(a, recipe.axis_atoms),)](
            bank,
            meta,
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
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _stamp("stream_forward", 1)
        _report("stream_forward", compiled)
        y = _restore(delta, y, "output")
        ctx.save_for_backward(meta, x, saved_w, origin, spacing, order, count)
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
        meta, x, w, origin, spacing, order, count = ctx.saved_tensors
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
        compiled = sk.backward[(tr.cdiv(a, recipe.axis_atoms),)](
            input_bank,
            adjoint_bank,
            meta,
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
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            num_warps=recipe.axis_warps,
            enable_fp_fusion=False,
        )
        _stamp("stream_backward", 1)
        _report("stream_backward", compiled)
        if ctx.needs_input_grad[0]:
            dx = _restore(delta, dx, "input")
        _stamp("unpack", 0)
        if ctx.needs_input_grad[1]:
            dp = torch.empty_like(dp).index_copy_(0, order, dp)
        _stamp("unpack", 1)
        return dx, dp, None, None, None, None, None, None


def apply(x, p, value, chart, recipe, bases, moments, buffers):
    key = (p.device, len(p), chart.grid_shape, recipe)
    if buffers.get("stream_layout_key") != key:
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        buffers["stream_layout"] = make_layout(len(p), sizes, recipe, p.device)
        buffers["stream_layout_key"] = key
    return _Streaming.apply(
        x.contiguous(),
        p,
        value,
        chart,
        recipe,
        bases,
        moments,
        buffers["stream_layout"],
    )
