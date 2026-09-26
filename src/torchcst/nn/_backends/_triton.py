"""Autograd and geometry preparation for the optional Triton backend.

Only O(N + K + atoms) prepared data crosses the kernel boundary; neither
weights nor their gradients are materialized as an N by K tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from torchcst.geometry import Chart, StripChart
from torchcst.kernels import Biweight, Kernel, Triangle, Triweight, WendlandC2
from torchcst.profiling import cst_span

from .._layout import initial_layout
from .._strip_torus import route_atoms
from ._torch import validate_tiled

if TYPE_CHECKING:
    from ..linear import CSTLinear

_PROFILES = {Biweight: 0, Triweight: 1, WendlandC2: 2, Triangle: 3}


def validate(charts: tuple[Chart, ...], kernel: Kernel) -> None:
    validate_tiled(charts, kernel)
    if type(kernel.profile) not in _PROFILES:
        raise ValueError(
            "triton backend requires Biweight, Triweight, WendlandC2 or Triangle"
        )


def geometry_factors(chart: StripChart) -> tuple[Tensor, Tensor]:
    """Factor ambient sites into row circle directions and column sections.

    For row n and column k, the site is
    [circle[n] * section[k, 0], section[k, 1:]]. Both tensors are linear
    in the operator's axis lengths, including for nonuniform column patterns.
    """

    geometry = chart.geometry
    rows = torch.arange(chart.shape[0], device=chart.device)
    arc = chart.axes[0].positions(rows % chart.tile_shape[0]).squeeze(-1)
    arc = (
        arc
        + torch.div(rows, chart.tile_shape[0], rounding_mode="floor") * chart.tile_pitch
    )
    angle = arc / geometry.major_radius
    circle = torch.stack((angle.cos(), angle.sin()), dim=-1)
    columns = torch.arange(chart.shape[1], device=chart.device)
    cross = chart.axes[1].positions(columns)
    section = torch.cat(
        (geometry.minor_radius.expand(cross.shape[0], 1), cross), dim=-1
    )
    section = section / torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    section = section * geometry.minor_radius
    section = torch.cat(
        (section[:, :1] + geometry.major_radius, section[:, 1:]), dim=-1
    )
    return circle.contiguous(), section.contiguous()


def forward(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
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

    with cst_span("cst.linear.route_atoms"):
        owners = route_atoms(site.chart, site.kernel, p)
        layout = initial_layout(owners, site.chart.tile_count)
    with cst_span("cst.linear.prepare_tiles"):
        center, amplitude, precision = site.kernel.tile_parameters(site.chart, p)
        center = site.chart.geometry.decode_centers(center)
        packed = layout.pack(
            torch.cat((amplitude[:, None], precision[:, None], center), dim=-1)
        )
        circle, section = geometry_factors(site.chart)
        flat = inputs.reshape(-1, site.in_features).contiguous()
    with cst_span("cst.linear.triton_matmul"):
        output = _FusedLinear.apply(
            flat,
            packed,
            circle,
            section,
            layout.offsets,
            site.chart.tile_shape[0],
            _PROFILES[type(site.kernel.profile)],
        )
    return output.reshape(*inputs.shape[:-1], site.out_features)


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class _FusedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, circle, section, offsets, station_rows, profile):
        from ._triton_kernels import fused_forward

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
        if inputs.shape[0]:
            grid = (
                _ceildiv(inputs.shape[0], bm),
                (offsets.numel() - 1) * _ceildiv(station_rows, bn),
            )
            with torch.cuda.device(inputs.device):
                fused_forward[grid](
                    inputs,
                    packed,
                    circle,
                    section,
                    offsets,
                    output,
                    **ctx.options,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        from ._triton_kernels import backward_atoms, backward_inputs

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
                    backward_inputs[grid](
                        gradient,
                        packed,
                        circle,
                        section,
                        offsets,
                        dx,
                        **ctx.options,
                        num_warps=4,
                        enable_fp_fusion=False,
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
        return dx, dp, None, None, None, None, None
