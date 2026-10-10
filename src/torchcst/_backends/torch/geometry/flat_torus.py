"""Flat periodic-coordinate geometry; no ambient doughnut embedding."""

import torch

from .base import _validate_center_structure, _validate_structure, validate_points
from .euclidean import project_tangent, transport  # noqa: F401


def wrap_delta(delta, periods):
    """Select [-L/2, L/2); at a half-period tie use the negative branch.

    The tie is a cut locus, so its derivative is this branch's one-sided VJP,
    not a claim of differentiability of shortest distance at the antipode.
    """
    return delta - periods * torch.floor(delta / periods + 0.5)


def center_offsets(self, sites, centers):
    _validate_structure(self, sites, name="sites")
    _validate_center_structure(self, centers, name="centers")
    return wrap_delta(sites[:, None, :] - centers[None, :, :], self.periods)


def squared_distance(self, sites, centers):
    return center_offsets(self, sites, centers).square().sum(-1)


def retract(self, points, displacement):
    validate_points(self, points, name="points")
    validate_points(self, displacement, name="displacement")
    if points.shape != displacement.shape:
        raise ValueError("points and displacement must have matching shapes")
    return torch.remainder(points + displacement, self.periods)


def initialize_centers(self, sites, atoms, *, mode):
    from .base import _validate_atoms
    from .euclidean import initialize_centers as euclidean_centers

    validate_points(self, sites, name="sites")
    _validate_atoms(atoms)
    if mode == "balanced":
        return euclidean_centers(self, sites, atoms, mode=mode)
    if mode != "uniform":
        raise ValueError("mode must be 'balanced' or 'uniform'")
    return (
        torch.rand(atoms, self.intrinsic_dim, device=sites.device, dtype=sites.dtype)
        * self.periods
    )
