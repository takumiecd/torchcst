"""Indexed regular-grid coordinates and bounded axis vectors."""

import torch

from torchcst.geometry.spec import FlatTorusGeometrySpec

from .lazy import _unravel


def positions(self, indices):
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or indices.ndim != 1
        or indices.device != self.device
    ):
        raise ValueError(
            "indices must be a one-dimensional long tensor on the chart device"
        )
    if bool(((indices < 0) | (indices >= self.features)).any()):
        raise IndexError("site index out of bounds")
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    grid_indices = torch.stack(_unravel(indices, self.coordinate_shape), dim=-1)
    return (
        self.origin.to(compute_dtype)
        + grid_indices.to(compute_dtype) * self.spacing.to(compute_dtype)
    ).to(self.dtype)


def initialize_centers(self, atoms, *, mode):
    if type(atoms) is not int or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    if mode == "uniform":
        compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        spacing = self.spacing.to(compute_dtype)
        extent = (
            self.geometry.periods.to(compute_dtype)
            if type(self.geometry.spec) is FlatTorusGeometrySpec
            else spacing * spacing.new_tensor([n - 1 for n in self.coordinate_shape])
        )
        return (
            self.origin.to(compute_dtype)
            + torch.rand(
                atoms, self.intrinsic_dim, device=self.device, dtype=compute_dtype
            )
            * extent
        ).to(self.dtype)
    if mode != "balanced":
        raise ValueError("mode must be 'balanced' or 'uniform'")
    indices = (
        torch.linspace(
            0, self.features - 1, atoms, device=self.device, dtype=torch.float64
        )
        .round()
        .long()
    )
    return positions(self, indices)


def _bounds(self):
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    spacing = self.spacing.to(compute_dtype)
    extent = spacing * spacing.new_tensor([n - 1 for n in self.coordinate_shape])
    return self.origin, (self.origin.to(compute_dtype) + extent).to(self.dtype)


def axis_positions(self):
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    for d, size in enumerate(self.coordinate_shape):
        yield (
            self.origin[d].to(compute_dtype)
            + torch.arange(size, device=self.device, dtype=compute_dtype)
            * self.spacing[d].to(compute_dtype)
        ).to(self.dtype)
