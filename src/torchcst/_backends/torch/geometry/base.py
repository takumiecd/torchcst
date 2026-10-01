"""PyTorch evaluation of base coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_points(self, points: Tensor, *, name: str = "points") -> None:
    """Validate an arbitrary table whose final axis stores one point."""

    self._validate_structure(points, name=name)
    if not bool(torch.isfinite(points).all()):
        raise ValueError(f"{name} must be finite")


def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
    """Validate stored center parameters."""

    self._validate_center_structure(centers, name=name)
    if not bool(torch.isfinite(centers).all()):
        raise ValueError(f"{name} must be finite")


def decode_centers(self, centers: Tensor) -> Tensor:
    """Decode stored centers into observation-site embedding coordinates."""

    self.validate_centers(centers)
    return centers


def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
    """Map lazy Chart coordinates into this geometry's observation space."""

    if (
        not isinstance(coordinates, Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[-1] != self.intrinsic_dim
        or not coordinates.is_floating_point()
    ):
        raise ValueError(f"coordinates must have shape [sites, {self.intrinsic_dim}]")
    if self.intrinsic_dim != self.embedding_dim:
        raise NotImplementedError("this geometry needs a chart-coordinate lift")
    return coordinates
