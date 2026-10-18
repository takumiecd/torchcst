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


def _adjoint_bank(dw, bank, sizes, start, recipe):
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


def _output_blocks(source, temporary, carry, sizes, start, recipe, adjoint):
    br, bc = recipe.scan_rows, recipe.scan_cols
    compiled = rk.output_scan_blocks[(tr.cdiv(sizes[0], br), tr.cdiv(sizes[1], bc))](
        source,
        temporary,
        carry,
        *sizes,
        start // 3,
        br,
        bc,
        adjoint,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("range_block_adjoint" if adjoint else "range_block_output", compiled)
    nb = tr.cdiv(sizes[0], br)
    compiled = rk.output_scan_carries[(tr.cdiv(sizes[1], 32),)](
        carry,
        sizes[1],
        nb,
        tr.next_power_of_2(nb),
        32,
        adjoint,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _report("range_block_carry", compiled)


def _matrix(meta, origin, spacing, count, sizes, recipe):
    w = _build_w(meta, origin, spacing, count, sizes, recipe, meta.device, "w_build")
    # Accumulate the coefficient groups in FP64 before the one FP32 W rounding.
    _stamp("range_build", 0)
    _stamp("range_init", 0)
    accum = w.double()
    delta = (
        torch.empty(
            (recipe.range_planes, *sizes), dtype=torch.float64, device=meta.device
        )
        if recipe.reuse_scratch
        else None
    )
    carry = (
        torch.empty(
            (tr.cdiv(sizes[0], recipe.scan_rows), sizes[1]),
            device=meta.device,
            dtype=torch.float64,
        )
        if recipe.blocked_scan
        else None
    )
    _stamp("range_init", 1)
    for start in range(0, 9, recipe.range_planes):
        _stamp(f"range_zero_{start}", 0)
        if recipe.reuse_scratch:
            delta.zero_()
        else:
            delta = torch.zeros(
                (recipe.range_planes, *sizes), dtype=torch.float64, device=meta.device
            )
        _stamp(f"range_zero_{start}", 1)
        _stamp(f"range_scatter_{start}", 0)
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
        _stamp(f"range_scatter_{start}", 1)
        if recipe.fused_scan:
            # A CTA loads every plane of its own row before storing that row.
            # Other CTAs own disjoint rows, so plane zero can hold the reduced scan.
            temp = (
                delta[0]
                if recipe.reuse_scratch
                else torch.empty(sizes, dtype=torch.float64, device=meta.device)
            )
            _stamp(f"range_scan_input_{start}", 0)
            compiled = rk.input_scan[(sizes[0],)](
                delta,
                temp,
                *sizes,
                tr.next_power_of_2(sizes[1]),
                num_warps=4,
                enable_fp_fusion=False,
            )
            _report("range_input_scan", compiled)
            _stamp(f"range_scan_input_{start}", 1)
            _stamp(f"range_scan_output_{start}", 0)
            if recipe.blocked_scan:
                _output_blocks(temp, temp, carry, sizes, start, recipe, False)
                compiled = rk.output_scan_restore[(tr.cdiv(sizes[0] * sizes[1], 256),)](
                    temp,
                    carry,
                    accum,
                    *sizes,
                    start // 3,
                    recipe.scan_rows,
                    256,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                compiled = rk.output_scan[(tr.cdiv(sizes[1], 4),)](
                    temp,
                    accum,
                    *sizes,
                    tr.next_power_of_2(sizes[0]),
                    start // 3,
                    4,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            _report("range_output_scan", compiled)
            _stamp(f"range_scan_output_{start}", 1)
            del temp
            if not recipe.reuse_scratch:
                del delta
            continue
        _stamp(f"range_scan_input_{start}", 0)
        _prefix(delta)
        _stamp(f"range_scan_input_{start}", 1)
        _stamp(f"range_scan_output_{start}", 0)
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
        _stamp(f"range_scan_output_{start}", 1)
        del delta
    _stamp("range_copy", 0)
    w.copy_(accum)
    _stamp("range_copy", 1)
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
            _stamp("range_partial_init", 0)
            partial = torch.zeros((len(meta), 3), device=dy.device, dtype=torch.float64)
            bank = (
                torch.empty(
                    (recipe.range_planes, *sizes), device=dy.device, dtype=torch.float64
                )
                if recipe.reuse_scratch
                else None
            )
            carry = (
                torch.empty(
                    (tr.cdiv(sizes[0], recipe.scan_rows), sizes[1]),
                    device=dy.device,
                    dtype=torch.float64,
                )
                if recipe.blocked_scan
                else None
            )
            _stamp("range_partial_init", 1)
            for start in range(0, 9, recipe.range_planes):
                if not recipe.reuse_scratch:
                    bank = torch.empty(
                        (recipe.range_planes, *sizes),
                        device=dy.device,
                        dtype=torch.float64,
                    )
                _stamp(f"range_adjoint_output_{start}", 0)
                if recipe.fused_scan:
                    # The input scan reads its whole row before overwriting plane zero.
                    temp = (
                        bank[0]
                        if recipe.reuse_scratch
                        else torch.empty(sizes, dtype=torch.float64, device=dy.device)
                    )
                    if recipe.blocked_scan:
                        _output_blocks(dw, temp, carry, sizes, start, recipe, True)
                    else:
                        compiled = rk.adjoint_output_scan[(tr.cdiv(sizes[1], 4),)](
                            dw,
                            temp,
                            *sizes,
                            tr.next_power_of_2(sizes[0]),
                            start // 3,
                            4,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                        _report("range_adjoint_output_scan", compiled)
                    _stamp(f"range_adjoint_output_{start}", 1)
                    _stamp(f"range_adjoint_input_{start}", 0)
                    if recipe.blocked_scan:
                        compiled = rk.adjoint_input_scan_carry[(sizes[0],)](
                            temp,
                            carry,
                            bank,
                            *sizes,
                            tr.next_power_of_2(sizes[1]),
                            recipe.scan_rows,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                    else:
                        compiled = rk.adjoint_input_scan[(sizes[0],)](
                            temp,
                            bank,
                            *sizes,
                            tr.next_power_of_2(sizes[1]),
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                    _report("range_adjoint_input_scan", compiled)
                    _stamp(f"range_adjoint_input_{start}", 1)
                    del temp
                else:
                    _adjoint_bank(dw, bank, sizes, start, recipe)
                    _stamp(f"range_adjoint_output_{start}", 1)
                    _stamp(f"range_adjoint_input_{start}", 0)
                    _stamp(f"range_adjoint_input_{start}", 1)
                _stamp(f"range_query_{start}", 0)
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
                    REVERSED=not recipe.fused_scan,
                    num_warps=recipe.range_warps,
                    enable_fp_fusion=False,
                )
                _report("range_vjp", compiled)
                _stamp(f"range_query_{start}", 1)
                if not recipe.reuse_scratch:
                    del bank
            _stamp("range_pullback", 0)
            rk.pullback[(tr.cdiv(len(meta), 128),)](
                meta, partial, packed, count, len(meta), 128
            )
            _stamp("range_pullback", 1)
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
