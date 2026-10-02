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

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.operators.dispatch import validate_tiled
from torchcst._backends.torch.operators.strip_torus.tiled import CircleRouting
from torchcst._backends.torch.patterns import execution as _patterns
from torchcst.charts import ChartState
from torchcst.kernels.state import KernelState

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear

PROFILE_KINDS = {"biweight": 0, "triweight": 1, "wendland_c2": 2, "triangle": 3}


def validate(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    validate_tiled(charts, kernel)
    if kernel.profiles[0].binding.profile.id not in PROFILE_KINDS:
        raise ValueError(
            "triton backend requires Biweight, Triweight, WendlandC2 or Triangle"
        )


def geometry_factors(chart: ChartState) -> tuple[Tensor, Tensor]:
    """Factor ambient sites into row circle directions and column sections.

    For row n and column k, the site is
    [circle[n] * section[k, 0], section[k, 1:]]. Both tensors are linear
    in the operator's axis lengths, including for nonuniform column patterns.
    """

    geometry = chart.geometry
    rows = torch.arange(chart.shape[0], device=chart.device)
    arc = _patterns.positions(chart.axes[0], rows % chart.tile_shape[0]).squeeze(-1)
    arc = (
        arc
        + torch.div(rows, chart.tile_shape[0], rounding_mode="floor") * chart.tile_pitch
    )
    angle = arc / geometry.major_radius
    circle = torch.stack((angle.cos(), angle.sin()), dim=-1)
    columns = torch.arange(chart.shape[1], device=chart.device)
    cross = _patterns.positions(chart.axes[1], columns)
    section = torch.cat(
        (geometry.minor_radius.expand(cross.shape[0], 1), cross), dim=-1
    )
    section = section / torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    section = section * geometry.minor_radius
    section = torch.cat(
        (section[:, :1] + geometry.major_radius, section[:, 1:]), dim=-1
    )
    return circle.contiguous(), section.contiguous()


def _configuration_key(chart: ChartState, kernel: KernelState) -> tuple | None:
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
        parameter_dim = _kernel.parameter_dim(site.kernel, site.chart)
        circle, section = geometry_factors(site.chart)
        plan = _Plan(
            key, parameter_dim, circle, section, CircleRouting.from_chart(site.chart)
        )
    site._strip_torus_plan = plan if key is not None else None
    return plan
