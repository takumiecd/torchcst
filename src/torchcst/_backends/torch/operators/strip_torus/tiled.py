"""PyTorch reference contracts for Strip + Torus tiled linear execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.operators.strip_torus.layout import AtomLayout
from torchcst._backends.torch.parameterizations import direct_amp_width as _coordinates
from torchcst.charts import ChartState, StripChartState
from torchcst.geometry.spec import TorusGeometrySpec
from torchcst.kernels import DirectAmpWidthSpec
from torchcst.kernels.profiles import (
    BiweightSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)
from torchcst.kernels.state import KernelState


def validate_tiled(chart: object, kernel: object) -> ChartState:
    """Check the deliberately narrow first tiled backend."""

    if (
        not (isinstance(chart, StripChartState))
        or not type(chart.geometry.spec) is TorusGeometrySpec
    ):
        raise TypeError("tiled backend requires StripChart + TorusGeometry")
    if (
        len(chart.shape) != 2
        or chart.axis != 0
        or chart.tile_shape[1] != chart.shape[1]
    ):
        raise ValueError(
            "tiled backend requires output-row stations and a full input axis"
        )
    if (
        not isinstance(kernel, KernelState)
        or type(kernel.spec.parameterization) is not DirectAmpWidthSpec
        or kernel.spec.composition != "radial"
        or type(kernel.profiles[0].binding.profile)
        not in (BiweightSpec, TriweightSpec, WendlandC2Spec, TriangleSpec)
        or kernel.profiles[0].binding.normalization.kind != "none"
    ):
        raise ValueError(
            "tiled backend requires DirectAmpWidth with a raw compact profile"
        )
    _kernel.family(kernel)
    kernel.declaration()
    if kernel.profiles[0].binding.profile.revision != 1:
        raise ValueError("unsupported profile revision")
    _charts.validate_support(chart, float(kernel.scalar("sigma_max_input")))
    return chart


@torch.no_grad()
def support_mask(chart: ChartState, kernel: KernelState, p: Tensor) -> Tensor:
    """Reference [station, atom] support oracle using actual site distances."""

    center = p[:, 2:]
    radius_squared = _coordinates.bandwidth_sigma(kernel, chart, p).square()
    touched = []
    for station in range(chart.tile_count):
        sites, _ = _charts.tile_indices(chart, station)
        squared = _charts.squared_distance(chart, center, sites)
        touched.append((squared < radius_squared[None, :]).any(dim=0))
    return torch.stack(touched)


@torch.no_grad()
def route_atoms(chart: ChartState, kernel: KernelState, p: Tensor) -> Tensor:
    """Own each atom by its nearest sampled row on the circle, in O(A * G).

    Every station has the same column cross-sections. At any fixed column,
    chord distance is minimized by the closest row angle, independently of
    the atom's section or bandwidth. Thus, if any site is supported, this
    owner has a supported site too. validate_support then guarantees that
    reading the owner and its two neighbors covers all contributions.
    """

    decoded = _geometry.decode_centers(chart.geometry, p[:, 2:])
    return CircleRouting.from_chart(chart).owners(decoded)


@dataclass(frozen=True)
class CircleRouting:
    """Fixed circle intervals, shared by reference and prepared execution."""

    major_radius: Tensor
    period: Tensor
    starts: Tensor
    spans: Tensor
    spacing: Tensor
    last_row: Tensor
    pitch: Tensor | None = None
    local_candidates: bool = False

    @classmethod
    def from_chart(cls, chart: ChartState) -> CircleRouting:
        stations = torch.arange(chart.tile_count, device=chart.device)
        line = chart.axes[chart.axis]
        counts = (
            chart.shape[chart.axis] - stations * chart.tile_shape[chart.axis]
        ).clamp_max(chart.tile_shape[chart.axis])
        # Ordered disjoint stations span less than one turn. In exact arithmetic
        # only the containing/predecessor and successor intervals can win, plus
        # the two endpoints across the circular gap. Keep two neighboring
        # stations on each side to absorb rounding in the candidate estimate.
        # This cold-plan guard bounds FP32 coordinate errors far below a pitch;
        # extreme coordinate scales keep the exhaustive reference path.
        start, pitch = float(line.start[0]), float(chart.tile_pitch)
        span = float(line.spacing[0]) * (chart.tile_shape[chart.axis] - 1)
        period = float(2 * torch.pi * chart.geometry.major_radius)
        last_span = float((counts[-1] - 1) * line.spacing[0])
        local_candidates = (
            chart.dtype == torch.float32
            and chart.tile_count >= 3
            and all(math.isfinite(v) for v in (start, pitch, span, period))
            and 0 <= span < pitch
            and (chart.tile_count - 1) * pitch + last_span < period
            and 256 * torch.finfo(torch.float32).eps * (abs(start) + period) < pitch
        )
        return cls(
            chart.geometry.major_radius,
            2 * torch.pi * chart.geometry.major_radius,
            line.start[0] + stations * chart.tile_pitch,
            (counts - 1) * line.spacing[0],
            line.spacing[0],
            counts - 1,
            chart.tile_pitch,
            local_candidates,
        )

    @torch.no_grad()
    def owners(self, decoded: Tensor) -> Tensor:
        arc = self.major_radius * torch.atan2(decoded[:, 1], decoded[:, 0])
        relative = (
            torch.remainder(
                arc[:, None] - (self.starts + self.spans / 2) + self.period / 2,
                self.period,
            )
            - self.period / 2
        )
        # A singleton/zero-spacing LinePattern has one sampled angle per station.
        spacing = self.spacing.clamp_min(torch.finfo(decoded.dtype).tiny)
        local = ((relative + self.spans / 2) / spacing).round().clamp_min(0)
        local = torch.minimum(local, self.last_row)
        nearest = self.starts + local * self.spacing
        distance = (
            torch.remainder(arc[:, None] - nearest + self.period / 2, self.period)
            - self.period / 2
        ).abs()
        return distance.argmin(dim=1)


@torch.no_grad()
def owners_from_support(chart: ChartState, p: Tensor, touched: Tensor) -> Tensor:
    """Assign one owner from an exact support mask."""

    if touched.shape != (chart.tile_count, p.shape[0]):
        raise ValueError("support mask has the wrong shape")
    owners = touched.to(torch.int64).argmax(dim=0)
    idle = ~touched.any(dim=0)
    if bool(idle.any()):
        centers = _geometry.decode_centers(chart.geometry, p[idle, 2:])
        center_angle = torch.atan2(centers[:, 1], centers[:, 0])
        midpoints = []
        for station in range(chart.tile_count):
            sites, _ = _charts.tile_indices(chart, station)
            endpoints = _charts.positions(chart, sites[[0, -1]])
            angle = torch.atan2(endpoints[:, 1], endpoints[:, 0])
            midpoints.append(torch.atan2(angle.sin().sum(), angle.cos().sum()))
        station_angle = torch.stack(midpoints)
        difference = center_angle[:, None] - station_angle[None, :]
        distance = torch.atan2(difference.sin(), difference.cos()).abs()
        owners[idle] = distance.argmin(dim=1)
    return owners


def tiled_linear(
    chart: ChartState,
    kernel: KernelState,
    inputs: Tensor,
    p: Tensor,
    layout: AtomLayout,
    *,
    support_layout: bool = False,
) -> Tensor:
    """Evaluate each chart-defined weight tile and multiply it immediately."""

    packed = layout.pack(p)
    input_features = chart.shape[1]
    flat_inputs = inputs.reshape(-1, input_features)
    outputs = []
    columns = torch.arange(input_features, device=p.device)
    for station in range(chart.tile_count):
        neighbors = dict.fromkeys(
            (
                (station - 1) % chart.tile_count,
                station,
                (station + 1) % chart.tile_count,
            )
        )
        if support_layout:
            from torchcst._backends.torch.operators.strip_torus.support import (
                station_buckets,
            )

            neighbors = station_buckets(station, chart.tile_count)
        candidates = torch.cat(
            [packed[layout.offsets[g] : layout.offsets[g + 1]] for g in neighbors]
        )
        row_start = station * chart.tile_shape[0]
        row_stop = min(row_start + chart.tile_shape[0], chart.shape[0])
        rows = torch.arange(row_start, row_stop, device=p.device)
        if candidates.shape[0]:
            weight = _kernel.weight_tile(kernel, chart, candidates, rows, columns)
            result = flat_inputs @ weight.T
        else:
            # Empty support still has zero derivatives with respect to X and
            # every atom, including when the entire operator is inactive.
            result = (flat_inputs.sum(-1, keepdim=True) * 0).expand(-1, rows.numel())
            result = result + packed.sum() * 0
        outputs.append(result)
    return torch.cat(outputs, dim=-1).reshape(*inputs.shape[:-1], chart.shape[0])
