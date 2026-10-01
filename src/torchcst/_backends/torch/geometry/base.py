"""PyTorch evaluation of base coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_points(self, points: Tensor, *, name: str = "points") -> None:
    """Validate an arbitrary table whose final axis stores one point."""

    _validate_structure(self, points, name=name)
    if not bool(torch.isfinite(points).all()):
        raise ValueError(f"{name} must be finite")


def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
    """Validate stored center parameters."""

    _validate_center_structure(self, centers, name=name)
    if not bool(torch.isfinite(centers).all()):
        raise ValueError(f"{name} must be finite")


def decode_centers(self, centers: Tensor) -> Tensor:
    """Decode stored centers into observation-site embedding coordinates."""

    validate_centers(self, centers)
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


def _validate_structure(state, points, *, name):
    from torch import Tensor

    if not isinstance(points, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if points.ndim < 1 or points.shape[-1] != state.embedding_dim:
        raise ValueError(f"{name} must have final dimension {state.embedding_dim}")
    if not points.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _validate_center_structure(state, centers, *, name):
    from torch import Tensor

    if not isinstance(centers, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if centers.ndim < 1 or centers.shape[-1] != state.center_parameter_dim:
        raise ValueError(
            f"{name} must have final dimension {state.center_parameter_dim}"
        )
    if not centers.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _validate_atoms(atoms):
    if type(atoms) is not int:
        raise TypeError("atoms must be an integer")
    if atoms < 1:
        raise ValueError("atoms must be positive")


def encode_centers(state, points):
    validate_points(state, points)
    return points
