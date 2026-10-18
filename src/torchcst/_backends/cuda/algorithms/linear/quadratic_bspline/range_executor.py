"""Batch-independent W assembly by 2D coefficient scans and adjoint VJP."""

import torch
import triton as tr
from torch.autograd.function import once_differentiable

from . import kernels as k
from . import range_kernels as rk
from .executor import _prepare, _report, _stamp
from .stream_executor import _build_w
from .stream_layout import make_layout, placement


def _prefix(delta):
    temporary = delta.cumsum(-1)
    torch.cumsum(temporary, -2, out=delta)
    return delta


def _matrix(meta, origin, spacing, count, sizes, recipe):
    w = _build_w(meta, origin, spacing, count, sizes, recipe, meta.device, "w_build")
    # Accumulate the coefficient groups in FP64 before the one FP32 W rounding.
    _stamp("range_build", 0)
    accum = w.double()
    for start in range(0, 9, recipe.range_planes):
        delta = torch.zeros(
            (recipe.range_planes, *sizes), dtype=torch.float64, device=meta.device
        )
        compiled = rk.scatter[(tr.cdiv(len(meta), recipe.range_atoms),)](
            meta,
            delta,
            count,
            origin,
            spacing,
            len(meta),
            *sizes,
            recipe.range_atoms,
            start,
            recipe.range_planes,
            recipe.reciprocal,
            num_warps=recipe.range_warps,
            enable_fp_fusion=False,
        )
        _report("range_build", compiled)
        _prefix(delta)
        rk.restore[(tr.cdiv(sizes[0] * sizes[1], 256),)](
            delta,
            accum,
            *sizes,
            start,
            recipe.range_planes,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        del delta
    w.copy_(accum)
    _stamp("range_build", 1)
    return w


class _Range(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, value, chart, recipe, moments, cache):
        sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
        ctx.recipe, ctx.sizes, ctx.empty = recipe, sizes, not len(p)
        if not len(p):
            ctx.save_for_backward(x)
            return x.new_zeros((len(x), sizes[0]))
        meta, origin, spacing, _ = _prepare(p, value, chart, recipe, moments)
        _stamp("group", 0)
        order, count = placement(meta, origin, spacing, sizes, recipe, cache)
        _stamp("group", 1)
        _stamp("pack", 0)
        meta = meta.index_select(0, order)
        _stamp("pack", 1)
        w = _matrix(meta, origin, spacing, count, sizes, recipe)
        _stamp("y_gemm", 0)
        y = x @ w.T
        _stamp("y_gemm", 1)
        saved = w if recipe.save_w else x.new_empty(0)
        ctx.save_for_backward(meta, x, saved, origin, spacing, order, count)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.empty:
            (x,) = ctx.saved_tensors
            return (
                torch.zeros_like(x) if ctx.needs_input_grad[0] else None,
                dy.new_empty((0, 4)) if ctx.needs_input_grad[1] else None,
                None,
                None,
                None,
                None,
                None,
            )
        meta, x, w, origin, spacing, order, count = ctx.saved_tensors
        recipe, sizes = ctx.recipe, ctx.sizes
        dy = dy.contiguous()
        dx = dp = None
        if ctx.needs_input_grad[0]:
            _stamp("w_rebuild", 0)
            if not recipe.save_w:
                w = _matrix(meta, origin, spacing, count, sizes, recipe)
            _stamp("w_rebuild", 1)
            _stamp("dx_gemm", 0)
            dx = dy @ w
            _stamp("dx_gemm", 1)
        if ctx.needs_input_grad[1]:
            _stamp("dw_gemm", 0)
            dw = dy.T @ x
            _stamp("dw_gemm", 1)
            packed = dy.new_empty((len(meta), 4))
            _stamp("matrix_atom_vjp", 0)
            compiled = k.matrix_vjp[(tr.cdiv(len(meta), recipe.matrix_atoms),)](
                meta,
                dw,
                packed,
                origin,
                spacing,
                len(meta),
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
            _stamp("matrix_atom_vjp", 1)
            _report("matrix_atom_vjp", compiled)
            _stamp("range_vjp", 0)
            partial = torch.zeros((len(meta), 3), device=dy.device, dtype=torch.float64)
            for start in range(0, 9, recipe.range_planes):
                bank = torch.empty(
                    (recipe.range_planes, *sizes), device=dy.device, dtype=torch.float64
                )
                rk.adjoint_init[(tr.cdiv(sizes[0] * sizes[1], 256),)](
                    dw,
                    bank,
                    *sizes,
                    start,
                    recipe.range_planes,
                    256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                _prefix(bank)
                compiled = rk.query[(tr.cdiv(len(meta), recipe.range_atoms),)](
                    bank,
                    meta,
                    partial,
                    count,
                    origin,
                    spacing,
                    len(meta),
                    *sizes,
                    recipe.range_atoms,
                    start,
                    recipe.range_planes,
                    recipe.reciprocal,
                    num_warps=recipe.range_warps,
                    enable_fp_fusion=False,
                )
                _report("range_vjp", compiled)
                del bank
            rk.pullback[(tr.cdiv(len(meta), 128),)](
                meta, partial, packed, count, len(meta), 128
            )
            _stamp("range_vjp", 1)
            _stamp("unpack", 0)
            dp = torch.empty_like(packed)
            dp.index_copy_(0, order, packed)
            _stamp("unpack", 1)
        return dx, dp, None, None, None, None, None


def apply(x, p, value, chart, recipe, moments, buffers):
    sizes = chart.grid_shape[0][0], chart.grid_shape[1][0]
    key = (x.device, len(p), sizes, recipe)
    if buffers.get("stream_layout_key") != key:
        buffers["stream_layout"] = make_layout(len(p), sizes, recipe, x.device)
        buffers["stream_layout_key"] = key
    return _Range.apply(
        x.contiguous(), p, value, chart, recipe, moments, buffers["stream_layout"]
    )
