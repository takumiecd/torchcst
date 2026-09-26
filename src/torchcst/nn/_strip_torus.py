"""PyTorch reference contracts for Strip + Torus tiled linear execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst.geometry import StripChart, TorusGeometry
from torchcst.kernels import DirectAmpWidth
from torchcst.kernels.compact import _CompactRadialProfile

from ._layout import AtomLayout


def validate_tiled(chart: object, kernel: object) -> StripChart:
    """Check the deliberately narrow first tiled backend."""

    if not isinstance(chart, StripChart) or not isinstance(
        chart.geometry, TorusGeometry
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
        not isinstance(kernel, DirectAmpWidth)
        or not isinstance(kernel.profile, _CompactRadialProfile)
        or kernel.profile.normalize_columns
    ):
        raise ValueError(
            "tiled backend requires DirectAmpWidth with a raw compact profile"
        )
    chart.validate_support(float(kernel.sigma_max))
    return chart


@torch.no_grad()
def support_mask(chart: StripChart, kernel: DirectAmpWidth, p: Tensor) -> Tensor:
    """Reference [station, atom] support oracle using actual site distances."""

    center = p[:, 2:]
    radius_squared = kernel.bandwidth_sigma(chart, p).square()
    touched = []
    for station in range(chart.tile_count):
        sites, _ = chart.tile_indices(station)
        squared = chart.squared_distance(center, sites)
        touched.append((squared < radius_squared[None, :]).any(dim=0))
    return torch.stack(touched)


@torch.no_grad()
def route_atoms(chart: StripChart, kernel: DirectAmpWidth, p: Tensor) -> Tensor:
    """Own each atom by its nearest sampled row on the circle, in O(A * G).

    Every station has the same column cross-sections. At any fixed column,
    chord distance is minimized by the closest row angle, independently of
    the atom's section or bandwidth. Thus, if any site is supported, this
    owner has a supported site too. validate_support then guarantees that
    reading the owner and its two neighbors covers all contributions.
    """

    decoded = chart.geometry.decode_centers(p[:, 2:])
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

    @classmethod
    def from_chart(cls, chart: StripChart) -> CircleRouting:
        stations = torch.arange(chart.tile_count, device=chart.device)
        line = chart.axes[chart.axis]
        counts = (
            chart.shape[chart.axis] - stations * chart.tile_shape[chart.axis]
        ).clamp_max(chart.tile_shape[chart.axis])
        return cls(
            chart.geometry.major_radius,
            2 * torch.pi * chart.geometry.major_radius,
            line.start[0] + stations * chart.tile_pitch,
            (counts - 1) * line.spacing[0],
            line.spacing[0],
            counts - 1,
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
def owners_from_support(chart: StripChart, p: Tensor, touched: Tensor) -> Tensor:
    """Assign one owner from an exact support mask."""

    if touched.shape != (chart.tile_count, p.shape[0]):
        raise ValueError("support mask has the wrong shape")
    owners = touched.to(torch.int64).argmax(dim=0)
    idle = ~touched.any(dim=0)
    if bool(idle.any()):
        centers = chart.geometry.decode_centers(p[idle, 2:])
        center_angle = torch.atan2(centers[:, 1], centers[:, 0])
        midpoints = []
        for station in range(chart.tile_count):
            sites, _ = chart.tile_indices(station)
            endpoints = chart.positions(sites[[0, -1]])
            angle = torch.atan2(endpoints[:, 1], endpoints[:, 0])
            midpoints.append(torch.atan2(angle.sin().sum(), angle.cos().sum()))
        station_angle = torch.stack(midpoints)
        difference = center_angle[:, None] - station_angle[None, :]
        distance = torch.atan2(difference.sin(), difference.cos()).abs()
        owners[idle] = distance.argmin(dim=1)
    return owners


def tiled_linear(
    chart: StripChart,
    kernel: DirectAmpWidth,
    inputs: Tensor,
    p: Tensor,
    layout: AtomLayout,
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
        candidates = torch.cat(
            [packed[layout.offsets[g] : layout.offsets[g + 1]] for g in neighbors]
        )
        row_start = station * chart.tile_shape[0]
        row_stop = min(row_start + chart.tile_shape[0], chart.shape[0])
        rows = torch.arange(row_start, row_stop, device=p.device)
        if candidates.shape[0]:
            weight = kernel.weight_tile(chart, candidates, rows, columns)
            result = flat_inputs @ weight.T
        else:
            result = flat_inputs.new_zeros((flat_inputs.shape[0], rows.numel()))
        outputs.append(result)
    return torch.cat(outputs, dim=-1).reshape(*inputs.shape[:-1], chart.shape[0])
