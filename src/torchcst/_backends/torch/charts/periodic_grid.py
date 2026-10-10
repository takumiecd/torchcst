"""Indexed periodic-grid coordinates without a whole-site table."""

import torch

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
    grid_indices = torch.stack(_unravel(indices, self.grid_shape), dim=-1)
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    origin, periods = (
        self.origin.to(compute_dtype),
        self.geometry.periods.to(compute_dtype),
    )
    spacing = periods / periods.new_tensor(self.grid_shape)
    return (origin + grid_indices.to(compute_dtype) * spacing).to(self.dtype)


def initialize_centers(self, atoms, *, mode):
    if type(atoms) is not int or atoms < 1:
        raise ValueError("atoms must be a positive integer")
    if mode == "uniform":
        return (
            self.origin
            + torch.rand(
                atoms, self.intrinsic_dim, device=self.device, dtype=self.dtype
            )
            * self.geometry.periods
        )
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
    return self.origin, self.origin + self.geometry.periods - self.spacing


def axis_positions(self):
    """Only O(sum n_d) axis coordinates; compute indices before narrowing."""
    compute_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
    for d, size in enumerate(self.grid_shape):
        yield (
            self.origin[d].to(compute_dtype)
            + torch.arange(size, device=self.device, dtype=compute_dtype)
            * (self.geometry.periods[d].to(compute_dtype) / size)
        ).to(self.dtype)
