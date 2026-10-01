"""PyTorch evaluation of sphere coordinate declarations."""

from __future__ import annotations

import torch
from torch import Tensor

from . import base
from .base import _validate_atoms, _validate_center_structure, _validate_structure


def validate_points(self, points: Tensor, *, name: str = "points") -> None:
    base.validate_points(self, points, name=name)
    radius = self.radius.to(points)
    norms = torch.linalg.vector_norm(points, dim=-1)
    # Charts may be constructed in float32 and promoted later. Keep a
    # representation-level tolerance instead of tightening it after a
    # dtype conversion that cannot recover the original precision.
    tolerance = radius.clamp_min(1) * 1e-5
    if not bool(torch.all((norms - radius).abs() <= tolerance)):
        raise ValueError(f"{name} must lie on the radius-{float(self.radius):g} sphere")


def _normalize(self, points: Tensor) -> Tensor:
    _validate_structure(self, points, name="points")
    norm = torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    tiny = torch.finfo(points.dtype).tiny
    return points * (self.radius.to(points) / norm.clamp_min(tiny))


def lift_tangent_sites(self, coordinates: Tensor) -> Tensor:
    """Map Cartesian axis coordinates to the northern sphere hemisphere.

    The gnomonic lift is injective and has unit local scale at the origin.
    This gives lazy product charts spherical sites without a site table.
    """

    if (
        not isinstance(coordinates, Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[-1] != self.intrinsic_dim
        or not coordinates.is_floating_point()
    ):
        raise ValueError(f"coordinates must have shape [sites, {self.intrinsic_dim}]")
    radius = self.radius.to(coordinates)
    denominator = torch.sqrt(
        radius.square() + coordinates.square().sum(-1, keepdim=True)
    )
    return (
        torch.cat(
            (radius.square().expand_as(denominator), radius * coordinates), dim=-1
        )
        / denominator
    )


def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
    return lift_tangent_sites(self, coordinates)


def max_parameter_radius(self) -> Tensor:
    """Largest intrinsic coordinate radius, excluding the antipodal cap."""

    return self.radius * (torch.pi - self.chart_margin)


def sample_sites(self, count: int) -> Tensor:
    """Sample ambient observation sites uniformly from the sphere."""

    _validate_atoms(count)
    samples = torch.randn(count, self.embedding_dim, device=self.radius.device)
    return _normalize(self, samples.to(dtype=self.radius.dtype))


def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
    base.validate_centers(self, centers, name=name)
    if self.representation == "ambient":
        radius = self.radius.to(centers)
        norms = torch.linalg.vector_norm(centers, dim=-1)
        tolerance = radius.clamp_min(1) * 1e-5
        if not bool(torch.all((norms - radius).abs() <= tolerance)):
            raise ValueError(
                f"{name} must lie on the radius-{float(self.radius):g} sphere"
            )
        return
    norms = torch.linalg.vector_norm(centers, dim=-1)
    limit = max_parameter_radius(self).to(centers)
    tolerance = limit.clamp_min(1) * 1e-5
    if not bool(torch.all(norms <= limit + tolerance)):
        raise ValueError(f"{name} must lie inside the intrinsic sphere chart")


def decode_centers(self, centers: Tensor) -> Tensor:
    _validate_center_structure(self, centers, name="centers")
    if self.representation == "ambient":
        return _normalize(self, centers)
    radius = self.radius.to(centers)
    radial = torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
    theta = radial / radius
    scale = torch.sinc(theta / torch.pi)
    north = radius * torch.cos(theta)
    return torch.cat((north, scale * centers), dim=-1)


def encode_centers(self, points: Tensor) -> Tensor:
    """Encode ambient sphere points in the configured center representation."""

    points = _normalize(self, points)
    if self.representation == "ambient":
        return points
    radius = self.radius.to(points)
    cosine = (points[..., :1] / radius).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    tail = points[..., 1:]
    tail_norm = torch.linalg.vector_norm(tail, dim=-1, keepdim=True)
    fallback = torch.zeros_like(tail)
    fallback[..., 0] = 1.0
    direction = torch.where(
        tail_norm > torch.finfo(points.dtype).eps,
        tail / tail_norm.clamp_min(torch.finfo(points.dtype).tiny),
        fallback,
    )
    encoded = radius * theta * direction
    return _clamp_intrinsic(self, encoded)


def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
    sites = _normalize(self, sites)
    centers = decode_centers(self, centers)
    return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)


def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
    sites = _normalize(self, sites)
    decoded = decode_centers(self, centers)
    offsets = sites[:, None, :] - decoded[None, :, :]
    if self.representation == "ambient":
        return project_tangent(self, centers, offsets)

    # Pull the ambient chord-distance derivative back through the sphere
    # exponential map. This is J_decode(center)^T @ (site - decoded).
    radius = self.radius.to(centers)
    radial = torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
    theta = radial / radius
    scale = torch.sinc(theta / torch.pi)
    cosine = torch.cos(theta)
    theta_square = theta.square()
    series = -1.0 / 3.0 + theta_square / 30.0 - theta_square.square() / 840.0
    curvature = torch.where(
        theta.abs() < 1e-3,
        series / radius.square(),
        (cosine - scale) / radial.square().clamp_min(torch.finfo(centers.dtype).tiny),
    )
    tangent_offset = offsets[..., 1:]
    radial_inner = (tangent_offset * centers[None, :, :]).sum(dim=-1, keepdim=True)
    return (
        scale[None, :, :] * tangent_offset
        - (scale / radius)[None, :, :] * centers[None, :, :] * offsets[..., :1]
        + curvature[None, :, :] * centers[None, :, :] * radial_inner
    )


def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
    validate_points(self, sites, name="sites")
    _validate_atoms(atoms)
    if mode == "balanced":
        indices = (
            torch.linspace(0, sites.shape[0] - 1, atoms, device=sites.device)
            .round()
            .to(dtype=torch.long)
        )
        return encode_centers(self, sites.index_select(0, indices))
    if mode != "uniform":
        raise ValueError("mode must be 'balanced' or 'uniform'")
    samples = torch.randn(
        atoms,
        self.embedding_dim,
        device=sites.device,
        dtype=sites.dtype,
    )
    return encode_centers(self, _normalize(self, samples))


def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
    if self.representation == "intrinsic":
        _validate_center_structure(self, points, name="points")
        _validate_center_structure(self, vectors, name="vectors")
        return vectors
    points = _normalize(self, points)
    _validate_structure(self, vectors, name="vectors")
    radial = (vectors * points).sum(dim=-1, keepdim=True)
    return vectors - radial * points / self.radius.to(points).square()


def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
    validate_centers(self, points, name="points")
    _validate_center_structure(self, displacement, name="displacement")
    if points.shape != displacement.shape:
        raise ValueError("points and displacement must have matching shapes")
    if self.representation == "intrinsic":
        return _clamp_intrinsic(self, points + displacement)
    tangent = project_tangent(self, points, displacement)
    return _normalize(self, points + tangent)


def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
    validate_centers(self, old, name="old")
    validate_centers(self, new, name="new")
    _validate_center_structure(self, vectors, name="vectors")
    if old.shape != new.shape or old.shape != vectors.shape:
        raise ValueError("old, new, and vectors must have matching shapes")
    if self.representation == "intrinsic":
        return vectors
    return project_tangent(self, new, vectors)


def _clamp_intrinsic(self, centers: Tensor) -> Tensor:
    _validate_center_structure(self, centers, name="centers")
    norm = torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
    limit = max_parameter_radius(self).to(centers)
    scale = (limit / norm.clamp_min(torch.finfo(centers.dtype).tiny)).clamp_max(1)
    return centers * scale
