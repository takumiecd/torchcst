"""PyTorch evaluation of euclidean coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry.geometry import (
    _validate_atoms,
)


def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
    self._validate_structure(sites, name="sites")
    self._validate_center_structure(centers, name="centers")
    return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)


def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
    self._validate_structure(sites, name="sites")
    self._validate_center_structure(centers, name="centers")
    return sites[:, None, :] - centers[None, :, :]


def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
    self.validate_points(sites, name="sites")
    _validate_atoms(atoms)
    if mode == "balanced":
        indices = (
            torch.linspace(0, sites.shape[0] - 1, atoms, device=sites.device)
            .round()
            .to(dtype=torch.long)
        )
        return sites.index_select(0, indices)
    if mode != "uniform":
        raise ValueError("mode must be 'balanced' or 'uniform'")
    low = sites.amin(dim=0)
    high = sites.amax(dim=0)
    unit = torch.rand(
        atoms,
        self.embedding_dim,
        device=sites.device,
        dtype=sites.dtype,
    )
    return low + unit * (high - low)


def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
    self._validate_structure(points, name="points")
    self._validate_structure(vectors, name="vectors")
    return vectors


def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
    self.validate_points(points, name="points")
    self.validate_points(displacement, name="displacement")
    if points.shape != displacement.shape:
        raise ValueError("points and displacement must have matching shapes")
    return points + displacement


def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
    self.validate_points(old, name="old")
    self.validate_points(new, name="new")
    self.validate_points(vectors, name="vectors")
    if old.shape != new.shape or old.shape != vectors.shape:
        raise ValueError("old, new, and vectors must have matching shapes")
    return vectors
