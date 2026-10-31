"""PyTorch evaluation of euclidean coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._builtin_callbacks import seal_functions as _seal_functions

from .base import (
    _validate_atoms,
    _validate_center_structure,
    _validate_structure,
    validate_points,
)


def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
    _validate_structure(self, sites, name="sites")
    _validate_center_structure(self, centers, name="centers")
    return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)


def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
    _validate_structure(self, sites, name="sites")
    _validate_center_structure(self, centers, name="centers")
    return sites[:, None, :] - centers[None, :, :]


def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
    validate_points(self, sites, name="sites")
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
    _validate_structure(self, points, name="points")
    _validate_structure(self, vectors, name="vectors")
    return vectors


def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
    validate_points(self, points, name="points")
    validate_points(self, displacement, name="displacement")
    if points.shape != displacement.shape:
        raise ValueError("points and displacement must have matching shapes")
    return points + displacement


def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
    from torchcst._backends.torch import finite

    names = ("old", "new", "vectors")
    values = (old, new, vectors)
    for name, value in zip(names, values):
        _validate_structure(self, value, name=name)
    bad = finite.first_nonfinite(values)
    if bad is not None:
        raise ValueError(f"{names[bad]} must be finite")
    if old.shape != new.shape or old.shape != vectors.shape:
        raise ValueError("old, new, and vectors must have matching shapes")
    return vectors


# Optional prepared updates may specialize only these original definitions.

_seal_functions(globals())
