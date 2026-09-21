"""Continuous geometries underlying fixed observation charts."""

from __future__ import annotations

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


def _validate_atoms(atoms: int) -> None:
    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an integer")
    if atoms < 1:
        raise ValueError("atoms must be positive")
