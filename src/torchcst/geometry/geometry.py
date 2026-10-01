"""Continuous geometries underlying fixed observation charts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from .spec import GeometrySpec


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

    def declaration(self) -> GeometrySpec:
        """Snapshot configuration, outside forward/backward and Graph capture.

        Device reads can synchronize; coordinate tables are copied into an
        immutable declaration. Custom implementations must override this.
        """
        from ._declarations import geometry_declaration

        return geometry_declaration(self)

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
        from torchcst._backends.torch.geometry.base import validate_points

        return validate_points(self, points, name=name)

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
        from torchcst._backends.torch.geometry.base import validate_centers

        return validate_centers(self, centers, name=name)

    def decode_centers(self, centers: Tensor) -> Tensor:
        """Decode stored centers into observation-site embedding coordinates."""
        from torchcst._backends.torch.geometry.base import decode_centers

        return decode_centers(self, centers)

    def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
        """Map lazy Chart coordinates into this geometry's observation space."""
        from torchcst._backends.torch.geometry.base import lift_chart_coordinates

        return lift_chart_coordinates(self, coordinates)

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
        from torchcst._backends.torch.geometry.euclidean import squared_distance

        return squared_distance(self, sites, centers)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.euclidean import center_offsets

        return center_offsets(self, sites, centers)

    def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
        from torchcst._backends.torch.geometry.euclidean import initialize_centers

        return initialize_centers(self, sites, atoms, mode=mode)

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.euclidean import project_tangent

        return project_tangent(self, points, vectors)

    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.euclidean import retract

        return retract(self, points, displacement)

    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.euclidean import transport

        return transport(self, old, new, vectors)


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
        from torchcst._backends.torch.geometry.sphere import validate_points

        return validate_points(self, points, name=name)

    def _normalize(self, points: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import _normalize

        return _normalize(self, points)

    def lift_tangent_sites(self, coordinates: Tensor) -> Tensor:
        """Map Cartesian axis coordinates to the northern sphere hemisphere.

        The gnomonic lift is injective and has unit local scale at the origin.
        This gives lazy product charts spherical sites without a site table.
        """
        from torchcst._backends.torch.geometry.sphere import lift_tangent_sites

        return lift_tangent_sites(self, coordinates)

    def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import lift_chart_coordinates

        return lift_chart_coordinates(self, coordinates)

    @property
    def max_parameter_radius(self) -> Tensor:
        """Largest intrinsic coordinate radius, excluding the antipodal cap."""
        from torchcst._backends.torch.geometry.sphere import max_parameter_radius

        return max_parameter_radius(self)

    def sample_sites(self, count: int) -> Tensor:
        """Sample ambient observation sites uniformly from the sphere."""
        from torchcst._backends.torch.geometry.sphere import sample_sites

        return sample_sites(self, count)

    def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
        from torchcst._backends.torch.geometry.sphere import validate_centers

        return validate_centers(self, centers, name=name)

    def decode_centers(self, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import decode_centers

        return decode_centers(self, centers)

    def encode_centers(self, points: Tensor) -> Tensor:
        """Encode ambient sphere points in the configured center representation."""
        from torchcst._backends.torch.geometry.sphere import encode_centers

        return encode_centers(self, points)

    def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import squared_distance

        return squared_distance(self, sites, centers)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import center_offsets

        return center_offsets(self, sites, centers)

    def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import initialize_centers

        return initialize_centers(self, sites, atoms, mode=mode)

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import project_tangent

        return project_tangent(self, points, vectors)

    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import retract

        return retract(self, points, displacement)

    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import transport

        return transport(self, old, new, vectors)

    def _clamp_intrinsic(self, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.sphere import _clamp_intrinsic

        return _clamp_intrinsic(self, centers)

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
    patch of the spherical cross-section. Centers can use ambient coordinates
    or an arc-length plus normal-coordinate representation. The kernel measures
    ambient chord distance, as in ``SphereGeometry``.
    """

    def __init__(
        self,
        intrinsic_dim: int,
        *,
        major_radius: float,
        minor_radius: float,
        circle_axis: int = 0,
        max_arc_step: float | None = None,
        representation: Literal["ambient", "intrinsic"] = "ambient",
        chart_margin: float = 0.05,
    ) -> None:
        if type(intrinsic_dim) is not int or intrinsic_dim < 2:
            raise ValueError("intrinsic_dim must be at least 2")
        if type(circle_axis) is not int or not 0 <= circle_axis < intrinsic_dim:
            raise ValueError("circle_axis must select one intrinsic coordinate")
        if representation not in ("ambient", "intrinsic"):
            raise ValueError("representation must be 'ambient' or 'intrinsic'")
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
        margin = torch.as_tensor(chart_margin, dtype=torch.get_default_dtype())
        if (
            margin.numel() != 1
            or not bool(torch.isfinite(margin))
            or margin <= 0
            or margin >= torch.pi
        ):
            raise ValueError("chart_margin must be finite and lie in (0, pi)")
        super().__init__(
            intrinsic_dim=intrinsic_dim,
            embedding_dim=intrinsic_dim + 1,
            center_parameter_dim=(
                intrinsic_dim + 1 if representation == "ambient" else intrinsic_dim
            ),
        )
        self.circle_axis = circle_axis
        self.max_arc_step = max_arc_step
        self.representation = representation
        self.register_buffer("major_radius", major.detach().clone().reshape(()))
        self.register_buffer("minor_radius", minor.detach().clone().reshape(()))
        self.register_buffer("chart_margin", margin.detach().clone().reshape(()))

    def _checkpoint_config(self) -> dict[str, object]:
        return {
            "circle_axis": self.circle_axis,
            "max_arc_step": self.max_arc_step,
            "representation": self.representation,
        }

    @property
    def circumference(self) -> float:
        from torchcst._backends.torch.geometry.torus import circumference

        return circumference(self)

    @property
    def max_section_parameter_radius(self) -> Tensor:
        """Normal-coordinate radius excluding the section's antipodal cap."""
        from torchcst._backends.torch.geometry.torus import max_section_parameter_radius

        return max_section_parameter_radius(self)

    def _embed(self, theta: Tensor, section: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import _embed

        return _embed(self, theta, section)

    def lift_chart_coordinates(self, coordinates: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import lift_chart_coordinates

        return lift_chart_coordinates(self, coordinates)

    def validate_points(self, points: Tensor, *, name: str = "points") -> None:
        from torchcst._backends.torch.geometry.torus import validate_points

        return validate_points(self, points, name=name)

    def validate_centers(self, centers: Tensor, *, name: str = "centers") -> None:
        from torchcst._backends.torch.geometry.torus import validate_centers

        return validate_centers(self, centers, name=name)

    def _decode_intrinsic(self, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import _decode_intrinsic

        return _decode_intrinsic(self, centers)

    def _encode_intrinsic(self, points: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import _encode_intrinsic

        return _encode_intrinsic(self, points)

    def encode_centers(self, points: Tensor) -> Tensor:
        """Encode ambient torus points in the selected center representation."""
        from torchcst._backends.torch.geometry.torus import encode_centers

        return encode_centers(self, points)

    def decode_centers(self, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import decode_centers

        return decode_centers(self, centers)

    def squared_distance(self, sites: Tensor, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import squared_distance

        return squared_distance(self, sites, centers)

    def _normal(self, points: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import _normal

        return _normal(self, points)

    def center_offsets(self, sites: Tensor, centers: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import center_offsets

        return center_offsets(self, sites, centers)

    def initialize_centers(self, sites: Tensor, atoms: int, *, mode: str) -> Tensor:
        from torchcst._backends.torch.geometry.torus import initialize_centers

        return initialize_centers(self, sites, atoms, mode=mode)

    def project_tangent(self, points: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import project_tangent

        return project_tangent(self, points, vectors)

    def _project_surface(self, points: Tensor, fallback: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import _project_surface

        return _project_surface(self, points, fallback)

    def retract(self, points: Tensor, displacement: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import retract

        return retract(self, points, displacement)

    def transport(self, old: Tensor, new: Tensor, vectors: Tensor) -> Tensor:
        from torchcst._backends.torch.geometry.torus import transport

        return transport(self, old, new, vectors)

    def axis_separation_lower_bound(self, gap: float) -> float:
        """Lower bound on chord distance for a wrapped major-circle gap."""
        from torchcst._backends.torch.geometry.torus import axis_separation_lower_bound

        return axis_separation_lower_bound(self, gap)

    def extra_repr(self) -> str:
        return (
            f"intrinsic_dim={self.intrinsic_dim}, "
            f"embedding_dim={self.embedding_dim}, "
            f"major_radius={float(self.major_radius):g}, "
            f"minor_radius={float(self.minor_radius):g}, "
            f"circle_axis={self.circle_axis}, max_arc_step={self.max_arc_step}, "
            f"representation={self.representation!r}"
        )


def _validate_atoms(atoms: int) -> None:
    if isinstance(atoms, bool) or not isinstance(atoms, int):
        raise TypeError("atoms must be an integer")
    if atoms < 1:
        raise ValueError("atoms must be positive")
