"""PyTorch evaluation of torus coordinate declarations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry.geometry import (
    TorusGeometry,
    _validate_atoms,
)


def circumference(self) -> float:
    return 2 * math.pi * float(self.major_radius)


def max_section_parameter_radius(self) -> Tensor:
    """Normal-coordinate radius excluding the section's antipodal cap."""

    return self.minor_radius * (torch.pi - self.chart_margin)


def _embed(self, theta: Tensor, section: Tensor) -> Tensor:
    major = self.major_radius.to(section)
    minor = self.minor_radius.to(section)
    radial = major + minor * section[..., :1]
    circle = radial * torch.stack((theta.cos(), theta.sin()), dim=-1)
    return torch.cat((circle, minor * section[..., 1:]), dim=-1)


def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
    if (
        not isinstance(coordinates, Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[-1] != self.intrinsic_dim
        or not coordinates.is_floating_point()
    ):
        raise ValueError(f"coordinates must have shape [sites, {self.intrinsic_dim}]")
    axis = coordinates[:, self.circle_axis]
    cross = torch.cat(
        (
            coordinates[:, : self.circle_axis],
            coordinates[:, self.circle_axis + 1 :],
        ),
        dim=-1,
    )
    minor = self.minor_radius.to(coordinates)
    section = torch.cat((minor.expand(cross.shape[0], 1), cross), dim=-1)
    section = section / torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    return self._embed(axis / self.major_radius.to(coordinates), section)


def validate_points(self, points: Tensor, *, name: str = "points") -> None:
    super(TorusGeometry, self).validate_points(points, name=name)
    radial = torch.linalg.vector_norm(points[..., :2], dim=-1)
    tube = torch.cat(
        ((radial - self.major_radius.to(points)).unsqueeze(-1), points[..., 2:]),
        dim=-1,
    )
    minor = self.minor_radius.to(points)
    # Recovering the tube radius subtracts two values on the major-radius
    # scale. Account for embedding/norm rounding at that scale, especially
    # for long Strip charts whose major radius grows with the tile count.
    coordinate_scale = torch.maximum(radial, self.major_radius.to(points))
    roundoff = 4 * torch.finfo(points.dtype).eps * coordinate_scale
    tolerance = minor.clamp_min(1) * 1e-5 + roundoff
    if not bool(
        torch.all((torch.linalg.vector_norm(tube, dim=-1) - minor).abs() <= tolerance)
    ):
        raise ValueError(f"{name} must lie on the torus surface")


def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
    self._validate_center_structure(centers, name=name)
    if self.representation == "ambient":
        self.validate_points(centers, name=name)
        return
    if not bool(torch.isfinite(centers).all()):
        raise ValueError(f"{name} must be finite")
    arc_limit = math.pi * float(self.major_radius)
    tolerance = max(1.0, arc_limit) * 1e-5
    if not bool(torch.all(centers[..., 0].abs() <= arc_limit + tolerance)):
        raise ValueError(f"{name} circle coordinate must lie within one turn")
    section_limit = self.max_section_parameter_radius.to(centers)
    if not bool(
        torch.all(
            torch.linalg.vector_norm(centers[..., 1:], dim=-1)
            <= section_limit + section_limit.clamp_min(1) * 1e-5
        )
    ):
        raise ValueError(f"{name} must lie inside the torus section chart")


def _decode_intrinsic(self, centers: Tensor) -> Tensor:
    radius = self.minor_radius.to(centers)
    section = centers[..., 1:]
    length = torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    angle = length / radius
    q = torch.cat(
        (
            angle.cos(),
            torch.sinc(angle / torch.pi) * section / radius,
        ),
        dim=-1,
    )
    return self._embed(centers[..., 0] / self.major_radius.to(centers), q)


def _encode_intrinsic(self, points: Tensor) -> Tensor:
    radial = torch.linalg.vector_norm(points[..., :2], dim=-1, keepdim=True)
    radius = self.minor_radius.to(points)
    angle = torch.acos(((radial - self.major_radius.to(points)) / radius).clamp(-1, 1))
    tail = points[..., 2:]
    tail_norm = torch.linalg.vector_norm(tail, dim=-1, keepdim=True)
    fallback = torch.zeros_like(tail)
    fallback[..., 0] = 1.0
    direction = torch.where(
        tail_norm > torch.finfo(points.dtype).eps,
        tail / tail_norm.clamp_min(torch.finfo(points.dtype).tiny),
        fallback,
    )
    section = radius * angle * direction
    section_norm = torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    limit = self.max_section_parameter_radius.to(points)
    section = section * (
        limit / section_norm.clamp_min(torch.finfo(points.dtype).tiny)
    ).clamp_max(1)
    arc = self.major_radius.to(points) * torch.atan2(points[..., 1:2], points[..., :1])
    return torch.cat((arc, section), dim=-1)


def encode_centers(self, points: Tensor) -> Tensor:
    """Encode ambient torus points in the selected center representation."""

    self.validate_points(points)
    if self.representation == "intrinsic":
        return self._encode_intrinsic(points)
    return self._project_surface(points, points)


def decode_centers(self, centers: Tensor) -> Tensor:
    self._validate_center_structure(centers, name="centers")
    if self.representation == "intrinsic":
        return self._decode_intrinsic(centers)
    return self._project_surface(centers, centers)


def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
    self._validate_structure(sites, name="sites")
    self._validate_center_structure(centers, name="centers")
    decoded = self.decode_centers(centers)
    return (sites[:, None, :] - decoded[None, :, :]).square().sum(dim=-1)


def _normal(self, points: Tensor) -> Tensor:
    radial = torch.linalg.vector_norm(points[..., :2], dim=-1, keepdim=True)
    direction = points[..., :2] / radial.clamp_min(torch.finfo(points.dtype).tiny)
    minor = self.minor_radius.to(points)
    section_radial = (radial - self.major_radius.to(points)) / minor
    return torch.cat((section_radial * direction, points[..., 2:] / minor), dim=-1)


def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
    self._validate_structure(sites, name="sites")
    self._validate_center_structure(centers, name="centers")
    decoded = self.decode_centers(centers)
    offsets = sites[:, None, :] - decoded[None, :, :]
    if self.representation == "intrinsic":
        radius = self.minor_radius.to(centers)
        major = self.major_radius.to(centers)
        angle = centers[:, 0] / major
        cosine, sine = angle.cos(), angle.sin()
        radial = torch.linalg.vector_norm(decoded[:, :2], dim=-1)
        circle_offset = (
            (-sine[None, :] * offsets[..., 0] + cosine[None, :] * offsets[..., 1])
            * radial[None, :]
            / major
        )
        section_radial_offset = (
            cosine[None, :] * offsets[..., 0] + sine[None, :] * offsets[..., 1]
        )[..., None]
        section_offset = offsets[..., 2:]
        section = centers[:, 1:]
        length = torch.linalg.vector_norm(section, dim=-1, keepdim=True)
        theta = length / radius
        scale = torch.sinc(theta / torch.pi)
        theta_square = theta.square()
        series = -1.0 / 3.0 + theta_square / 30.0 - theta_square.square() / 840.0
        curvature = torch.where(
            theta.abs() < 1e-3,
            series / radius.square(),
            (theta.cos() - scale)
            / length.square().clamp_min(torch.finfo(centers.dtype).tiny),
        )
        radial_inner = (section_offset * section[None, :, :]).sum(dim=-1, keepdim=True)
        section_gradient = (
            scale[None, :, :] * section_offset
            - (scale / radius)[None, :, :] * section[None, :, :] * section_radial_offset
            + curvature[None, :, :] * section[None, :, :] * radial_inner
        )
        return torch.cat((circle_offset[..., None], section_gradient), dim=-1)
    normal = self._normal(decoded)[None, :, :]
    return offsets - (offsets * normal).sum(dim=-1, keepdim=True) * normal


def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
    self.validate_points(sites, name="sites")
    _validate_atoms(atoms)
    if mode == "balanced":
        indices = (
            torch.linspace(0, sites.shape[0] - 1, atoms, device=sites.device)
            .round()
            .long()
        )
        selected = sites.index_select(0, indices)
        return self.encode_centers(selected)
    if mode != "uniform":
        raise ValueError("mode must be 'balanced' or 'uniform'")
    theta = 2 * torch.pi * torch.rand(atoms, device=sites.device, dtype=sites.dtype)
    section = torch.randn(
        atoms, self.intrinsic_dim, device=sites.device, dtype=sites.dtype
    )
    section = section / torch.linalg.vector_norm(section, dim=-1, keepdim=True)
    sampled = self._embed(theta, section)
    return self.encode_centers(sampled)


def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
    if self.representation == "intrinsic":
        self._validate_center_structure(points, name="points")
        self._validate_center_structure(vectors, name="vectors")
        return vectors
    self._validate_structure(points, name="points")
    self._validate_structure(vectors, name="vectors")
    normal = self._normal(points)
    return vectors - (vectors * normal).sum(dim=-1, keepdim=True) * normal


def _project_surface(self, points: Tensor, fallback: Tensor) -> Tensor:
    radial = torch.linalg.vector_norm(points[..., :2], dim=-1, keepdim=True)
    safe = torch.finfo(points.dtype).eps
    fallback_radial = torch.linalg.vector_norm(fallback[..., :2], dim=-1, keepdim=True)
    default_circle = torch.cat(
        (torch.ones_like(radial), torch.zeros_like(radial)), dim=-1
    )
    fallback_circle = torch.where(
        fallback_radial > safe,
        fallback[..., :2] / fallback_radial.clamp_min(safe),
        default_circle,
    )
    circle = torch.where(
        radial > safe,
        points[..., :2] / radial.clamp_min(safe),
        fallback_circle,
    )
    tube = torch.cat(((radial - self.major_radius.to(points)), points[..., 2:]), dim=-1)
    tube_norm = torch.linalg.vector_norm(tube, dim=-1, keepdim=True)
    old_radial = fallback_radial - self.major_radius.to(fallback)
    old_tube = torch.cat((old_radial, fallback[..., 2:]), dim=-1)
    old_tube_norm = torch.linalg.vector_norm(old_tube, dim=-1, keepdim=True)
    default_section = torch.cat(
        (torch.ones_like(tube_norm), torch.zeros_like(tube[..., 1:])), dim=-1
    )
    fallback_section = torch.where(
        old_tube_norm > safe,
        old_tube / old_tube_norm.clamp_min(safe),
        default_section,
    )
    section = torch.where(
        tube_norm > safe,
        tube / tube_norm.clamp_min(safe),
        fallback_section,
    )
    minor = self.minor_radius.to(points)
    ring_radius = self.major_radius.to(points) + minor * section[..., :1]
    return torch.cat((ring_radius * circle, minor * section[..., 1:]), dim=-1)


def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
    self.validate_centers(points, name="points")
    self._validate_center_structure(displacement, name="displacement")
    if points.shape != displacement.shape:
        raise ValueError("points and displacement must have matching shapes")
    if not bool(torch.isfinite(displacement).all()):
        raise ValueError("displacement must be finite")
    if self.representation == "intrinsic":
        arc_step = displacement[..., :1]
        if self.max_arc_step is not None:
            arc_step = arc_step.clamp(-self.max_arc_step, self.max_arc_step)
        half_turn = math.pi * float(self.major_radius)
        arc = (
            torch.remainder(points[..., :1] + arc_step + half_turn, 2 * half_turn)
            - half_turn
        )
        section = points[..., 1:] + displacement[..., 1:]
        length = torch.linalg.vector_norm(section, dim=-1, keepdim=True)
        limit = self.max_section_parameter_radius.to(section)
        section = section * (
            limit / length.clamp_min(torch.finfo(section.dtype).tiny)
        ).clamp_max(1)
        return torch.cat((arc, section), dim=-1)
    tangent = self.project_tangent(points, displacement)
    updated = self._project_surface(points + tangent, points)
    if self.max_arc_step is None:
        return updated
    old_circle = points[..., :2]
    new_circle = updated[..., :2]
    cross = (
        old_circle[..., 0] * new_circle[..., 1]
        - old_circle[..., 1] * new_circle[..., 0]
    )
    dot = (old_circle * new_circle).sum(dim=-1)
    turn = torch.atan2(cross, dot)
    angle_limit = self.max_arc_step / float(self.major_radius)
    limited = turn.clamp(-angle_limit, angle_limit)
    old_angle = torch.atan2(old_circle[..., 1], old_circle[..., 0])
    radius = torch.linalg.vector_norm(new_circle, dim=-1)
    circle = radius[..., None] * torch.stack(
        ((old_angle + limited).cos(), (old_angle + limited).sin()), dim=-1
    )
    return torch.cat((circle, updated[..., 2:]), dim=-1)


def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
    self.validate_centers(old, name="old")
    self.validate_centers(new, name="new")
    self._validate_center_structure(vectors, name="vectors")
    if old.shape != new.shape or old.shape != vectors.shape:
        raise ValueError("old, new, and vectors must have matching shapes")
    if self.representation == "intrinsic":
        return vectors
    return self.project_tangent(new, vectors)


def axis_separation_lower_bound(self, gap: float) -> float:
    """Lower bound on chord distance for a wrapped major-circle gap."""

    if not 0 <= gap <= self.circumference / 2:
        raise ValueError("gap must lie in the first half of the major circle")
    return (
        2
        * (float(self.major_radius) - float(self.minor_radius))
        * math.sin(gap / (2 * float(self.major_radius)))
    )
