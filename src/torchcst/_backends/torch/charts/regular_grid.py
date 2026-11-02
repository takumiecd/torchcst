"""Indexed regular-grid coordinates and bounded axis vectors."""

import math

import torch

from torchcst.geometry.spec import EuclideanGeometrySpec, FlatTorusGeometrySpec

from .lazy import _unravel


def positions(self, indices, *, like=None):
    device = self.device if like is None else like.device
    dtype = self.dtype if like is None else like.dtype
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or indices.ndim != 1
        or indices.device != device
    ):
        raise ValueError(
            "indices must be a one-dimensional long tensor on the chart device"
        )
    if bool(((indices < 0) | (indices >= self.features)).any()):
        raise IndexError("site index out of bounds")
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    grid_indices = torch.stack(_unravel(indices, self.coordinate_shape), dim=-1)
    limit = torch.finfo(dtype).max
    if any(
        not math.isfinite(o)
        or not math.isfinite(s)
        or s <= 0
        or abs(o) > limit
        or abs(s) > torch.finfo(compute_dtype).max
        or abs(o + (n - 1) * s) > limit
        for o, s, n in zip(self.origin, self.spacing, self.coordinate_shape)
    ):
        raise ValueError("grid positions are not finite at the requested dtype")
    result = (
        torch.tensor(self.origin, device=device, dtype=compute_dtype)
        + grid_indices.to(compute_dtype)
        * torch.tensor(self.spacing, device=device, dtype=compute_dtype)
    ).to(dtype)
    return result


def initialize_centers(self, atoms, *, mode):
    if type(atoms) is not int or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    if mode == "uniform":
        compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        spacing = torch.tensor(self.spacing, device=self.device, dtype=compute_dtype)
        extent = (
            spacing.new_tensor(self.geometry.periods)
            if type(self.geometry.spec) is FlatTorusGeometrySpec
            else spacing * spacing.new_tensor([n - 1 for n in self.coordinate_shape])
        )
        return (
            torch.tensor(self.origin, device=self.device, dtype=compute_dtype)
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
    spacing = torch.tensor(self.spacing, device=self.device, dtype=compute_dtype)
    extent = spacing * spacing.new_tensor([n - 1 for n in self.coordinate_shape])
    return torch.tensor(self.origin, device=self.device, dtype=self.dtype), (
        torch.tensor(self.origin, device=self.device, dtype=compute_dtype) + extent
    ).to(self.dtype)


def axis_positions(self):
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    for d, size in enumerate(self.coordinate_shape):
        yield (
            torch.tensor(self.origin[d], device=self.device, dtype=compute_dtype)
            + torch.arange(size, device=self.device, dtype=compute_dtype)
            * torch.tensor(self.spacing[d], device=self.device, dtype=compute_dtype)
        ).to(self.dtype)


def squared_distance(self, centers, selection=None):
    return _distance(self, centers, selection, offsets=False)


def center_offsets(self, centers, selection=None):
    return _distance(self, centers, selection, offsets=True)


def _distance(self, centers, selection, *, offsets):
    from torchcst._backends.torch.geometry import execution as geometry

    if selection is None:
        indices = torch.arange(self.features, device=centers.device)
    elif isinstance(selection, slice):
        indices = torch.arange(*selection.indices(self.features), device=centers.device)
    else:
        indices = selection
    sites = positions(self, indices, like=centers)
    if type(self.geometry.spec) is not EuclideanGeometrySpec:
        geometry.validate_points(self.geometry, sites, name="sites")
    operation = geometry.center_offsets if offsets else geometry.squared_distance
    return operation(self.geometry, sites, centers)
