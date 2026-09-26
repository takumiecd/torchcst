"""Reusable fixed geometry and fresh per-forward atom preparation.

The plan never caches trainable atom values, sort orders or autograd graphs.
Buffer versions and checkpoint metadata invalidate fixed geometry after .to(),
load_state_dict(), in-place buffer updates or configuration replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from torchcst.geometry import Chart, StripChart
from torchcst.kernels import (
    Biweight,
    DirectAmpWidth,
    Kernel,
    Triangle,
    Triweight,
    WendlandC2,
)
from torchcst.profiling import cst_span

from .._layout import _station_layout
from .._strip_torus import CircleRouting
from ._torch import validate_tiled

if TYPE_CHECKING:
    from ..linear import CSTLinear

PROFILE_KINDS = {Biweight: 0, Triweight: 1, WendlandC2: 2, Triangle: 3}


def validate(charts: tuple[Chart, ...], kernel: Kernel) -> None:
    validate_tiled(charts, kernel)
    if type(kernel.profile) not in PROFILE_KINDS:
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


def _configuration_key(chart: StripChart, kernel: DirectAmpWidth) -> tuple | None:
    parts = []
    for root in (chart, kernel):
        for name, module in root.named_modules():
            metadata = (
                module.get_extra_state()
                if type(module).get_extra_state is not nn.Module.get_extra_state
                else None
            )
            parts.append((name, module, metadata))
        for name, buffer in root.named_buffers():
            # Inference-created tensors have no version counter. Rebuild rather
            # than reuse a plan whose mutations cannot be detected.
            if buffer.is_inference():
                return None
            parts.append(
                (
                    name,
                    id(buffer),
                    buffer._version,
                    buffer.device,
                    buffer.dtype,
                    buffer.shape,
                )
            )
    return tuple(parts)


@dataclass(frozen=True)
class _Plan:
    key: tuple | None
    parameter_dim: int
    circle: Tensor
    section: Tensor
    routing: CircleRouting


def execution_plan(site: CSTLinear) -> _Plan:
    key = _configuration_key(site.chart, site.kernel)
    previous = getattr(site, "_strip_torus_plan", None)
    if key is not None and previous is not None and previous.key == key:
        return previous
    # Constants created during inference must still be normal tensors if the
    # same model is subsequently trained and autograd saves them for backward.
    with torch.inference_mode(False), torch.no_grad():
        validate(site.cst_charts(), site.kernel)
        parameter_dim = site.kernel.parameter_dim(site.chart)
        circle, section = geometry_factors(site.chart)
        plan = _Plan(
            key, parameter_dim, circle, section, CircleRouting.from_chart(site.chart)
        )
    site._strip_torus_plan = plan if key is not None else None
    return plan


def prepare(site: CSTLinear, p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    plan = execution_plan(site)
    if p.ndim != 2 or p.shape[1] != plan.parameter_dim:
        raise ValueError(f"p must have shape [atoms, {plan.parameter_dim}]")
    with cst_span("cst.linear.prepare_atoms"):
        # Preserve overrides on custom kernels; the built-in evaluator can
        # reuse the plan's configuration validation without GPU/CPU sync.
        if type(site.kernel).tile_parameters is DirectAmpWidth.tile_parameters:
            center, amplitude, precision = site.kernel._tile_parameters(p)
        else:
            center, amplitude, precision = site.kernel.tile_parameters(site.chart, p)
        decoded = site.chart.geometry.decode_centers(center)
    with cst_span("cst.linear.route_atoms"):
        owners = plan.routing.owners(decoded)
        layout = _station_layout(owners, site.chart.tile_count)
    with cst_span("cst.linear.pack_atoms"):
        packed = layout.pack(
            torch.cat((amplitude[:, None], precision[:, None], decoded), dim=-1)
        )
    return packed, plan.circle, plan.section, layout.offsets
