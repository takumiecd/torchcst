"""Continuous geometries underlying fixed observation charts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Geometry(nn.Module, ABC):
    """Metric and update geometry for chart sites and atom centers.

    ``embedding_dim`` is the stored coordinate width. ``intrinsic_dim`` is
    the number of geometric degrees of freedom. They agree in Euclidean
    space and differ for embedded manifolds such as ``SphereGeometry``.
    """

    def __init__(self, *, intrinsic_dim: int, embedding_dim: int) -> None:
        super().__init__()
        for name, value in (
            ("intrinsic_dim", intrinsic_dim),
            ("embedding_dim", embedding_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.intrinsic_dim = intrinsic_dim
        self.embedding_dim = embedding_dim

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
        self._validate_structure(centers, name="centers")
        return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        self._validate_structure(sites, name="sites")
        self._validate_structure(centers, name="centers")
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

    Distances are squared ambient chord distances. Updates are tangent-projected
    and normalized back to ``radius``; no coordinate chart or angular singularity
    is exposed to kernels or optimizers.
    """

    def __init__(self, intrinsic_dim: int, *, radius: float = 1.0) -> None:
        super().__init__(
            intrinsic_dim=intrinsic_dim,
            embedding_dim=intrinsic_dim + 1,
        )
        value = torch.as_tensor(radius, dtype=torch.get_default_dtype())
        if value.numel() != 1 or not bool(torch.isfinite(value)) or value <= 0:
            raise ValueError("radius must be finite and positive")
        self.register_buffer("radius", value.detach().clone().reshape(()))

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

    def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
        sites = self._normalize(sites)
        centers = self._normalize(centers)
        return (sites[:, None, :] - centers[None, :, :]).square().sum(dim=-1)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        sites = self._normalize(sites)
        centers = self._normalize(centers)
        offsets = sites[:, None, :] - centers[None, :, :]
        return self.project_tangent(centers, offsets)

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
        samples = torch.randn(
            atoms,
            self.embedding_dim,
            device=sites.device,
            dtype=sites.dtype,
        )
        return self._normalize(samples)

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        points = self._normalize(points)
        self._validate_structure(vectors, name="vectors")
        radial = (vectors * points).sum(dim=-1, keepdim=True)
        return vectors - radial * points / self.radius.to(points).square()

    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        self.validate_points(points, name="points")
        Geometry.validate_points(self, displacement, name="displacement")
        if points.shape != displacement.shape:
            raise ValueError("points and displacement must have matching shapes")
        tangent = self.project_tangent(points, displacement)
        return self._normalize(points + tangent)

    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        self.validate_points(old, name="old")
        self.validate_points(new, name="new")
        Geometry.validate_points(self, vectors, name="vectors")
        if old.shape != new.shape or old.shape != vectors.shape:
            raise ValueError("old, new, and vectors must have matching shapes")
        return self.project_tangent(new, vectors)

    def extra_repr(self) -> str:
        return (
            f"intrinsic_dim={self.intrinsic_dim}, "
            f"embedding_dim={self.embedding_dim}, radius={float(self.radius):g}"
        )


def _validate_atoms(atoms: int) -> None:
    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an integer")
    if atoms < 1:
        raise ValueError("atoms must be positive")
