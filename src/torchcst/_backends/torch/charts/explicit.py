"""PyTorch evaluation of explicit coordinate declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from torchcst.geometry.chart import Chart
from torchcst.geometry.geometry import Geometry, SphereGeometry


def points(
    cls,
    coordinates: Tensor,
    *,
    geometry: Geometry | None = None,
    trainable: bool = False,
) -> Chart:
    """Construct a chart from an explicit ``[features, dimensions]`` tensor."""

    return cls(coordinates, geometry=geometry, trainable=trainable)


def sphere(
    cls,
    features: int,
    *,
    intrinsic_dim: int,
    radius: float = 1.0,
    representation: Literal["ambient", "intrinsic"] = "ambient",
    chart_margin: float = 0.05,
    trainable: bool = False,
) -> Chart:
    """Construct points sampled uniformly on the intrinsic sphere ``S^d``."""

    cls._validate_size(features, name="features")
    geometry = SphereGeometry(
        intrinsic_dim,
        radius=radius,
        representation=representation,
        chart_margin=chart_margin,
    )
    coordinates = geometry.sample_sites(features)
    return cls(coordinates, geometry=geometry, trainable=trainable)


def linspace(
    cls,
    size: int,
    *,
    spacing: float | None = None,
    low: float | None = None,
    high: float | None = None,
    center: float | None = None,
    trainable: bool = False,
) -> Chart:
    """Construct a one-dimensional evenly spaced chart."""

    cls._validate_size(size, name="size")
    axis, step = cls._axis(
        size,
        spacing=spacing,
        low=low,
        high=high,
        center=center,
        name="linspace",
    )
    chart = cls(axis.unsqueeze(-1), trainable=trainable)
    chart._set_spacing((step,) if step is not None else None)
    return chart


def grid(
    cls,
    shape: Sequence[int],
    *,
    spacing: float | Sequence[float] | None = None,
    low: float | None = None,
    high: float | None = None,
    center: float | Sequence[float] | None = None,
    trainable: bool = False,
) -> Chart:
    """Construct a Cartesian grid with one coordinate axis per shape entry."""

    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError("shape must be a non-empty sequence of integers")
    shape = tuple(shape)
    if not shape:
        raise ValueError("shape must contain at least one dimension")
    for index, size in enumerate(shape):
        cls._validate_size(size, name=f"shape[{index}]")

    dim = len(shape)
    steps = cls._per_axis_values(spacing, dim=dim, name="spacing", allow_none=True)
    centers = cls._per_axis_values(center, dim=dim, name="center", allow_none=True)
    if steps is not None and (low is not None or high is not None):
        raise ValueError("specify spacing or low/high, not both")
    if (low is None) ^ (high is None):
        raise ValueError("low and high must be passed together")
    if steps is None and low is None:
        raise ValueError("specify spacing or low and high")
    if low is not None and center is not None:
        raise ValueError("center is only used with spacing")

    axes = []
    stored_steps: list[float] = []
    for index, size in enumerate(shape):
        axis, step = cls._axis(
            size,
            spacing=None if steps is None else steps[index],
            low=low,
            high=high,
            center=None if centers is None else centers[index],
            name=f"grid[{index}]",
        )
        axes.append(axis)
        if step is not None:
            stored_steps.append(step)
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, dim
    )
    chart = cls(coordinates, trainable=trainable)
    chart._set_spacing(tuple(stored_steps) if len(stored_steps) == dim else None)
    return chart


def _axis(
    cls,
    size: int,
    *,
    spacing: float | None,
    low: float | None,
    high: float | None,
    center: float | None,
    name: str,
) -> tuple[Tensor, float | None]:
    if spacing is not None and (low is not None or high is not None):
        raise ValueError("specify spacing or low/high, not both")
    if (low is None) ^ (high is None):
        raise ValueError("low and high must be passed together")
    if spacing is None and low is None:
        raise ValueError("specify spacing or low and high")
    if spacing is not None:
        step = cls._positive_float(spacing, name="spacing")
        origin = 0.0 if center is None else cls._finite_float(center, name="center")
        index = torch.arange(size, dtype=torch.get_default_dtype())
        axis = (index - (size - 1) / 2) * step + origin
        return axis, step
    if center is not None:
        raise ValueError("center is only used with spacing")
    start = cls._finite_float(low, name="low")
    end = cls._finite_float(high, name="high")
    if size == 1:
        return torch.tensor([start], dtype=torch.get_default_dtype()), None
    if end < start:
        raise ValueError("high must not be smaller than low")
    axis = torch.linspace(start, end, size)
    return axis, float(axis[1] - axis[0])


def positions(self, indices: Tensor) -> Tensor:
    """Return coordinates for requested flattened site indices."""

    return self.coordinates.index_select(0, indices)


def squared_distance(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    """Pairwise site-center squared distance in this chart's geometry."""

    sites = self.coordinates if selection is None else self.coordinates[selection]
    return self.geometry.squared_distance(sites, centers)


def center_offsets(
    self, centers: Tensor, selection: slice | Tensor | None = None
) -> Tensor:
    """Site-center offsets in each center's tangent space."""

    sites = self.coordinates if selection is None else self.coordinates[selection]
    return self.geometry.center_offsets(sites, centers)


def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
    """Initialize atom centers in the chart's geometry."""

    return self.geometry.initialize_centers(self.coordinates, atoms, mode=mode)
