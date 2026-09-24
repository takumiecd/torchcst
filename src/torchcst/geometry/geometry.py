"""Continuous geometries underlying fixed observation charts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn


class Geometry(nn.Module, ABC):
    """Metric and update geometry for chart sites and atom centers.

    ``embedding_dim`` is the coordinate width of observation sites.
    ``center_parameter_dim`` is the stored width of an atom center, while
    ``intrinsic_dim`` is its number of geometric degrees of freedom.
    """

    def __init__(
        self,
        *,
        intrinsic_dim: int,
        embedding_dim: int,
        center_parameter_dim: int | None = None,
    ) -> None:
        super().__init__()
        if center_parameter_dim is None:
            center_parameter_dim = embedding_dim
        for name, value in (
            ("intrinsic_dim", intrinsic_dim),
            ("embedding_dim", embedding_dim),
            ("center_parameter_dim", center_parameter_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.intrinsic_dim = intrinsic_dim
        self.embedding_dim = embedding_dim
        self.center_parameter_dim = center_parameter_dim

    def get_extra_state(self) -> dict[str, object]:
        """Record non-tensor settings that define center-coordinate meaning."""

        return {
            "format_version": 1,
            "geometry_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "intrinsic_dim": self.intrinsic_dim,
            "embedding_dim": self.embedding_dim,
            "center_parameter_dim": self.center_parameter_dim,
            "config": self._checkpoint_config(),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError(
                "geometry checkpoint contract differs from this geometry"
            )

    def _checkpoint_config(self) -> dict[str, object]:
        """Subclass-specific non-tensor settings; scalar buffers load separately."""

        return {}

    def validate_points(self, points: Tensor, *, name: str = "points") -> None:
        """Validate an arbitrary table whose final axis stores one point."""

        self._validate_structure(points, name=name)
        if not bool(torch.isfinite(points).all()):
            raise ValueError(f"{name} must be finite")

    def _validate_structure(self, points: Tensor, *, name: str) -> None:
        """Validate metadata without tensor-to-bool control flow."""

        if not isinstance(points, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if points.ndim < 1 or points.shape[-1] != self.embedding_dim:
            raise ValueError(f"{name} must have final dimension {self.embedding_dim}")
        if not points.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")

    def _validate_center_structure(self, centers: Tensor, *, name: str) -> None:
        if not isinstance(centers, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if centers.ndim < 1 or centers.shape[-1] != self.center_parameter_dim:
            raise ValueError(
                f"{name} must have final dimension {self.center_parameter_dim}"
            )
        if not centers.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype")

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
            raise ValueError(
                f"coordinates must have shape [sites, {self.intrinsic_dim}]"
            )
        if self.intrinsic_dim != self.embedding_dim:
            raise NotImplementedError("this geometry needs a chart-coordinate lift")
        return coordinates

    @abstractmethod
    def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
        """Return pairwise squared distances with shape ``[sites, atoms]``."""

    @abstractmethod
    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        """Return center-tangent offsets with shape ``[sites, atoms, storage]``."""

    @abstractmethod
    def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
        """Initialize one stored center coordinate per atom."""

    @abstractmethod
    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        """Project ambient vectors into the tangent space at ``points``."""

    @abstractmethod
    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        """Apply a tangent displacement and return valid geometry points."""

    @abstractmethod
    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        """Move tangent vectors from ``old`` to the tangent space at ``new``."""


class EuclideanGeometry(Geometry):
    """Ordinary Euclidean geometry with unconstrained additive updates."""

    def __init__(self, dim: int) -> None:
        super().__init__(intrinsic_dim=dim, embedding_dim=dim)

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


class SphereGeometry(Geometry):
    r"""The intrinsic ``S^d`` sphere embedded in ``R^(d+1)``.

    Observation sites always use the ambient ``R^(d+1)`` representation.
    Centers may either use that robust redundant representation or ``d`` normal
    coordinates around the north pole. Distances are squared ambient chord
    distances after decoding, so the two center representations have the same
    semantics away from the intrinsic chart's excluded antipodal cap.
    """

    def __init__(
        self,
        intrinsic_dim: int,
        *,
        radius: float = 1.0,
        representation: Literal["ambient", "intrinsic"] = "ambient",
        chart_margin: float = 0.05,
    ) -> None:
        if representation not in ("ambient", "intrinsic"):
            raise ValueError("representation must be 'ambient' or 'intrinsic'")
        super().__init__(
            intrinsic_dim=intrinsic_dim,
            embedding_dim=intrinsic_dim + 1,
            center_parameter_dim=(
                intrinsic_dim + 1 if representation == "ambient" else intrinsic_dim
            ),
        )
        value = torch.as_tensor(radius, dtype=torch.get_default_dtype())
        if value.numel() != 1 or not bool(torch.isfinite(value)) or value <= 0:
            raise ValueError("radius must be finite and positive")
        margin = torch.as_tensor(chart_margin, dtype=torch.get_default_dtype())
        if (
            margin.numel() != 1
            or not bool(torch.isfinite(margin))
            or margin <= 0
            or margin >= torch.pi
        ):
            raise ValueError("chart_margin must be finite and lie in (0, pi)")
        self.representation = representation
        self.register_buffer("radius", value.detach().clone().reshape(()))
        self.register_buffer("chart_margin", margin.detach().clone().reshape(()))

    def _checkpoint_config(self) -> dict[str, object]:
        return {"representation": self.representation}

    def validate_points(self, points: Tensor, *, name: str = "points") -> None:
        super().validate_points(points, name=name)
        radius = self.radius.to(points)
        norms = torch.linalg.vector_norm(points, dim=-1)
        # Charts may be constructed in float32 and promoted later. Keep a
        # representation-level tolerance instead of tightening it after a
        # dtype conversion that cannot recover the original precision.
        tolerance = radius.clamp_min(1) * 1e-5
        if not bool(torch.all((norms - radius).abs() <= tolerance)):
            raise ValueError(
                f"{name} must lie on the radius-{float(self.radius):g} sphere"
            )

    def _normalize(self, points: Tensor) -> Tensor:
        self._validate_structure(points, name="points")
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
            raise ValueError(
                f"coordinates must have shape [sites, {self.intrinsic_dim}]"
            )
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
        return self.lift_tangent_sites(coordinates)

    @property
    def max_parameter_radius(self) -> Tensor:
        """Largest intrinsic coordinate radius, excluding the antipodal cap."""

        return self.radius * (torch.pi - self.chart_margin)

    def sample_sites(self, count: int) -> Tensor:
        """Sample ambient observation sites uniformly from the sphere."""

        _validate_atoms(count)
        samples = torch.randn(count, self.embedding_dim, device=self.radius.device)
        return self._normalize(samples.to(dtype=self.radius.dtype))

    def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
        super().validate_centers(centers, name=name)
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
        limit = self.max_parameter_radius.to(centers)
        tolerance = limit.clamp_min(1) * 1e-5
        if not bool(torch.all(norms <= limit + tolerance)):
            raise ValueError(f"{name} must lie inside the intrinsic sphere chart")

    def decode_centers(self, centers: Tensor) -> Tensor:
        self._validate_center_structure(centers, name="centers")
        if self.representation == "ambient":
            return self._normalize(centers)
        radius = self.radius.to(centers)
        radial = torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
        theta = radial / radius
        scale = torch.sinc(theta / torch.pi)
        north = radius * torch.cos(theta)
        return torch.cat((north, scale * centers), dim=-1)

    def encode_centers(self, points: Tensor) -> Tensor:
        """Encode ambient sphere points in the configured center representation."""

        points = self._normalize(points)
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
        return self._clamp_intrinsic(encoded)

    def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
        sites = self._normalize(sites)
        centers = self.decode_centers(centers)
        return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        sites = self._normalize(sites)
        decoded = self.decode_centers(centers)
        offsets = sites[:, None, :] - decoded[None, :, :]
        if self.representation == "ambient":
            return self.project_tangent(centers, offsets)

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
            (cosine - scale)
            / radial.square().clamp_min(torch.finfo(centers.dtype).tiny),
        )
        tangent_offset = offsets[..., 1:]
        radial_inner = (tangent_offset * centers[None, :, :]).sum(dim=-1, keepdim=True)
        return (
            scale[None, :, :] * tangent_offset
            - (scale / radius)[None, :, :] * centers[None, :, :] * offsets[..., :1]
            + curvature[None, :, :] * centers[None, :, :] * radial_inner
        )

    def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
        self.validate_points(sites, name="sites")
        _validate_atoms(atoms)
        if mode == "balanced":
            indices = (
                torch.linspace(0, sites.shape[0] - 1, atoms, device=sites.device)
                .round()
                .to(dtype=torch.long)
            )
            return self.encode_centers(sites.index_select(0, indices))
        if mode != "uniform":
            raise ValueError("mode must be 'balanced' or 'uniform'")
        samples = torch.randn(
            atoms,
            self.embedding_dim,
            device=sites.device,
            dtype=sites.dtype,
        )
        return self.encode_centers(self._normalize(samples))

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        if self.representation == "intrinsic":
            self._validate_center_structure(points, name="points")
            self._validate_center_structure(vectors, name="vectors")
            return vectors
        points = self._normalize(points)
        self._validate_structure(vectors, name="vectors")
        radial = (vectors * points).sum(dim=-1, keepdim=True)
        return vectors - radial * points / self.radius.to(points).square()

    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        self.validate_centers(points, name="points")
        self._validate_center_structure(displacement, name="displacement")
        if points.shape != displacement.shape:
            raise ValueError("points and displacement must have matching shapes")
        if self.representation == "intrinsic":
            return self._clamp_intrinsic(points + displacement)
        tangent = self.project_tangent(points, displacement)
        return self._normalize(points + tangent)

    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        self.validate_centers(old, name="old")
        self.validate_centers(new, name="new")
        self._validate_center_structure(vectors, name="vectors")
        if old.shape != new.shape or old.shape != vectors.shape:
            raise ValueError("old, new, and vectors must have matching shapes")
        if self.representation == "intrinsic":
            return vectors
        return self.project_tangent(new, vectors)

    def _clamp_intrinsic(self, centers: Tensor) -> Tensor:
        self._validate_center_structure(centers, name="centers")
        norm = torch.linalg.vector_norm(centers, dim=-1, keepdim=True)
        limit = self.max_parameter_radius.to(centers)
        scale = (limit / norm.clamp_min(torch.finfo(centers.dtype).tiny)).clamp_max(1)
        return centers * scale

    def extra_repr(self) -> str:
        return (
            f"intrinsic_dim={self.intrinsic_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"center_parameter_dim={self.center_parameter_dim}, "
            f"representation={self.representation!r}, radius={float(self.radius):g}"
        )


class TorusGeometry(Geometry):
    r"""The hypersurface ``S^1 × S^(d-1)`` embedded in ``R^(d+1)``.

    One Chart coordinate follows the major circle, measured as arc length at
    ``major_radius``. The remaining ``d-1`` coordinates lift to a northern
    patch of the spherical cross-section. Centers use ambient coordinates and
    the kernel measures ambient chord distance, as in ``SphereGeometry``.
    """

    def __init__(
        self,
        intrinsic_dim: int,
        *,
        major_radius: float,
        minor_radius: float,
        circle_axis: int = 0,
        max_arc_step: float | None = None,
    ) -> None:
        if type(intrinsic_dim) is not int or intrinsic_dim < 2:
            raise ValueError("intrinsic_dim must be at least 2")
        if type(circle_axis) is not int or not 0 <= circle_axis < intrinsic_dim:
            raise ValueError("circle_axis must select one intrinsic coordinate")
        major = torch.as_tensor(major_radius, dtype=torch.get_default_dtype())
        minor = torch.as_tensor(minor_radius, dtype=torch.get_default_dtype())
        if (
            major.numel() != 1
            or minor.numel() != 1
            or not bool(torch.isfinite(major))
            or not bool(torch.isfinite(minor))
            or minor <= 0
            or major <= minor
        ):
            raise ValueError("radii must be finite and satisfy major > minor > 0")
        if max_arc_step is not None and (
            not math.isfinite(max_arc_step)
            or max_arc_step <= 0
            or max_arc_step > math.pi * float(major)
        ):
            raise ValueError("max_arc_step must lie in (0, pi * major_radius]")
        super().__init__(intrinsic_dim=intrinsic_dim, embedding_dim=intrinsic_dim + 1)
        self.circle_axis = circle_axis
        self.max_arc_step = max_arc_step
        self.register_buffer("major_radius", major.detach().clone().reshape(()))
        self.register_buffer("minor_radius", minor.detach().clone().reshape(()))

    def _checkpoint_config(self) -> dict[str, object]:
        return {"circle_axis": self.circle_axis, "max_arc_step": self.max_arc_step}

    @property
    def circumference(self) -> float:
        return 2 * math.pi * float(self.major_radius)

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
            raise ValueError(
                f"coordinates must have shape [sites, {self.intrinsic_dim}]"
            )
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
        super().validate_points(points, name=name)
        radial = torch.linalg.vector_norm(points[..., :2], dim=-1)
        tube = torch.cat(
            ((radial - self.major_radius.to(points)).unsqueeze(-1), points[..., 2:]),
            dim=-1,
        )
        minor = self.minor_radius.to(points)
        tolerance = minor.clamp_min(1) * 1e-5
        if not bool(
            torch.all(
                (torch.linalg.vector_norm(tube, dim=-1) - minor).abs() <= tolerance
            )
        ):
            raise ValueError(f"{name} must lie on the torus surface")

    def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
        self._validate_center_structure(centers, name=name)
        self.validate_points(centers, name=name)

    def decode_centers(self, centers: Tensor) -> Tensor:
        self._validate_center_structure(centers, name="centers")
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
            return sites.index_select(0, indices)
        if mode != "uniform":
            raise ValueError("mode must be 'balanced' or 'uniform'")
        theta = 2 * torch.pi * torch.rand(atoms, device=sites.device, dtype=sites.dtype)
        section = torch.randn(
            atoms, self.intrinsic_dim, device=sites.device, dtype=sites.dtype
        )
        section = section / torch.linalg.vector_norm(section, dim=-1, keepdim=True)
        return self._embed(theta, section)

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        self._validate_structure(points, name="points")
        self._validate_structure(vectors, name="vectors")
        normal = self._normal(points)
        return vectors - (vectors * normal).sum(dim=-1, keepdim=True) * normal

    def _project_surface(self, points: Tensor, fallback: Tensor) -> Tensor:
        radial = torch.linalg.vector_norm(points[..., :2], dim=-1, keepdim=True)
        safe = torch.finfo(points.dtype).eps
        fallback_radial = torch.linalg.vector_norm(
            fallback[..., :2], dim=-1, keepdim=True
        )
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
        tube = torch.cat(
            ((radial - self.major_radius.to(points)), points[..., 2:]), dim=-1
        )
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

    def extra_repr(self) -> str:
        return (
            f"intrinsic_dim={self.intrinsic_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"major_radius={float(self.major_radius):g}, "
            f"minor_radius={float(self.minor_radius):g}, "
            f"circle_axis={self.circle_axis}, max_arc_step={self.max_arc_step}"
        )


def _validate_atoms(atoms: int) -> None:
    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an integer")
    if atoms < 1:
        raise ValueError("atoms must be positive")
