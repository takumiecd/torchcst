"""Autograd and geometry preparation for the optional Triton backend.

Only O(N + K + atoms) prepared data crosses the kernel boundary; neither
weights nor their gradients are materialized as an N by K tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from torchcst.profiling import cst_span

from ._preparation import PROFILE_KINDS, prepare
from ._schedule import split_count

if TYPE_CHECKING:
    from ..linear import CSTLinear


def forward(
    site: CSTLinear, inputs: Tensor, p: Tensor, *, split_reductions: bool = False
) -> Tensor:
    if inputs.device.type != "cuda" or p.device != inputs.device or torch.version.hip:
        raise ValueError(
            "triton backend requires inputs and atoms on the same NVIDIA CUDA device"
        )
    if inputs.dtype != torch.float32 or p.dtype != torch.float32:
        raise TypeError("the first triton backend requires float32 inputs and atoms")
    if site.chart.device != p.device or site.chart.dtype != p.dtype:
        raise ValueError(
            "triton backend requires chart and atoms to share device and dtype"
        )
    try:
        from . import _triton_kernels  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "triton backend requires the optional torchcst[cuda] dependencies"
        ) from error

    packed, circle, section, offsets = prepare(site, p)
    flat = inputs.reshape(-1, site.in_features).contiguous()
    with cst_span("cst.linear.triton_matmul"):
        output = _FusedLinear.apply(
            flat,
            packed,
            circle,
            section,
            offsets,
            site.chart.tile_shape[0],
            PROFILE_KINDS[type(site.kernel.profile)],
            split_reductions,
        )
    return output.reshape(*inputs.shape[:-1], site.out_features)


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class _FusedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        packed,
        circle,
        section,
        offsets,
        station_rows,
        profile,
        split_reductions=False,
    ):
        from ._triton_kernels import fused_forward, reduce_partials

        output = torch.empty(
            (inputs.shape[0], circle.shape[0]), device=inputs.device, dtype=inputs.dtype
        )
        ctx.save_for_backward(inputs, packed, circle, section, offsets)
        ctx.options = {
            "M": inputs.shape[0],
            "N": circle.shape[0],
            "K": inputs.shape[1],
            "D": packed.shape[1] - 2,
            "G": offsets.numel() - 1,
            "STATION_ROWS": station_rows,
            "PROFILE": profile,
            "BM": 16,
            "BN": 16,
            "BK": 16,
            "BA": 8,
        }
        bm, bn = ctx.options["BM"], ctx.options["BN"]
        # Keep the production path unchanged until the extra allocation and
        # launch overhead also improves ordinary (uncaptured) execution.
        ctx.multiprocessors = 0
        if split_reductions:
            properties = torch.cuda.get_device_properties(inputs.device)
            if "A100" in properties.name:
                ctx.multiprocessors = properties.multi_processor_count
        if inputs.shape[0]:
            grid = (
                _ceildiv(inputs.shape[0], bm),
                (offsets.numel() - 1) * _ceildiv(station_rows, bn),
            )
            parts = split_count(
                reduction_tiles=_ceildiv(inputs.shape[1], ctx.options["BK"]),
                elements=output.numel(),
                programs=grid[0] * grid[1],
                multiprocessors=ctx.multiprocessors,
                atoms=packed.shape[0],
                stations=offsets.numel() - 1,
            )
            partial = output if parts == 1 else output.new_empty((parts, *output.shape))
            with torch.cuda.device(inputs.device):
                fused_forward[(*grid, parts)](
                    inputs,
                    packed,
                    circle,
                    section,
                    offsets,
                    partial,
                    **ctx.options,
                    SPLIT_K=parts,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                if parts > 1:
                    reduce_partials[(_ceildiv(output.numel(), 256),)](
                        partial, output, output.numel(), parts, 256
                    )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        from ._triton_kernels import backward_atoms, backward_inputs, reduce_partials

        inputs, packed, circle, section, offsets = ctx.saved_tensors
        gradient = output_gradient.contiguous()
        dx = torch.empty_like(inputs) if ctx.needs_input_grad[0] else None
        dp = torch.zeros_like(packed) if ctx.needs_input_grad[1] else None
        if (
            dp is not None
            and inputs.shape[0]
            and torch.are_deterministic_algorithms_enabled()
        ):
            raise RuntimeError(
                "triton atom backward uses atomic accumulation; use backend='tiled' "
                "for deterministic gradients"
            )
        bm, bn, bk = (ctx.options[name] for name in ("BM", "BN", "BK"))
        if inputs.shape[0]:
            with torch.cuda.device(inputs.device):
                if dx is not None:
                    grid = (
                        _ceildiv(inputs.shape[0], bm),
                        _ceildiv(inputs.shape[1], bk),
                    )
                    parts = split_count(
                        reduction_tiles=offsets.numel() - 1,
                        elements=dx.numel(),
                        programs=grid[0] * grid[1],
                        multiprocessors=ctx.multiprocessors,
                        atoms=packed.shape[0],
                        stations=offsets.numel() - 1,
                    )
                    partial = dx if parts == 1 else dx.new_empty((parts, *dx.shape))
                    backward_inputs[(*grid, parts)](
                        gradient,
                        packed,
                        circle,
                        section,
                        offsets,
                        partial,
                        **ctx.options,
                        SPLIT_N=parts,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
                    if parts > 1:
                        reduce_partials[(_ceildiv(dx.numel(), 256),)](
                            partial, dx, dx.numel(), parts, 256
                        )
                if dp is not None:
                    grid = (
                        (offsets.numel() - 1)
                        * _ceildiv(ctx.options["STATION_ROWS"], bn),
                        _ceildiv(inputs.shape[1], bk),
                    )
                    backward_atoms[grid](
                        inputs,
                        gradient,
                        packed,
                        circle,
                        section,
                        offsets,
                        dp,
                        **ctx.options,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
        return (dx, dp) + (None,) * (len(ctx.needs_input_grad) - 2)
