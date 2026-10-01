"""PyTorch evaluation of base coordinate declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from torch import Tensor

from torchcst.geometry.chart import Chart, ExplicitChart
from torchcst.geometry.geometry import Geometry


def points(
    cls,
    coordinates: Tensor,
    *,
    geometry: Geometry | None = None,
    trainable: bool = False,
) -> Chart:
    return ExplicitChart.points(coordinates, geometry=geometry, trainable=trainable)


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
    return ExplicitChart.linspace(
        size,
        spacing=spacing,
        low=low,
        high=high,
        center=center,
        trainable=trainable,
    )


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
    return ExplicitChart.grid(
        shape,
        spacing=spacing,
        low=low,
        high=high,
        center=center,
        trainable=trainable,
    )


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
    return ExplicitChart.sphere(
        features,
        intrinsic_dim=intrinsic_dim,
        radius=radius,
        representation=representation,
        chart_margin=chart_margin,
        trainable=trainable,
    )
