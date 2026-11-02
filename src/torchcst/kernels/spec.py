"""Immutable mathematical declarations, independent of backend selection."""

import math
from dataclasses import dataclass, field
from typing import Literal

from .normalization import NormalizationSpec
from .parameterizations import (
    FixedWidthSpec,
    ParameterizationSpec,
    PhysicalAmpRadiusSpec,
    PolarAmpRadiusSpec,
)
from .profiles import ProfileSpec

AtomInit = Literal["balanced", "uniform"]


@dataclass(frozen=True, kw_only=True)
class ProfileBinding:
    """A profile shape, default bandwidth and explicit normalization."""

    profile: ProfileSpec
    parameterization: FixedWidthSpec | None = None
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)

    def __post_init__(self):
        if (
            not isinstance(self.profile, ProfileSpec)
            or (
                self.parameterization is not None
                and not isinstance(self.parameterization, FixedWidthSpec)
            )
            or not isinstance(self.normalization, NormalizationSpec)
        ):
            raise TypeError("invalid profile binding")

        # Existing bindings evaluate profiles at points. They cannot silently
        # interpret the new integral rule as sampled L2 normalization.
        if self.normalization.kind == "continuous_mass":
            raise ValueError(
                "continuous mass normalization requires a compatible cell-integrated "
                "profile contract; existing ProfileBinding does not support it"
            )


@dataclass(frozen=True, kw_only=True)
class RadiusIntegratedProfileBinding(ProfileBinding):
    """Whole-cell masses with physical support radius and integer hierarchy.

    Local full periods are P=(ceil(2R/T)+1)*T. Base tiles and merge factor
    change the mathematical approximation and belong to this declaration.
    """

    base_tile_sites: int = 1
    merge_factor: int = 2
    normalization: NormalizationSpec = field(
        default_factory=lambda: NormalizationSpec(
            kind="continuous_mass", domain="profile_support"
        )
    )

    def __post_init__(self):
        from .profiles.fourier import ChebyshevRadiusSpec

        if (
            type(self.profile) is not ChebyshevRadiusSpec
            or self.parameterization is not None
        ):
            raise ValueError(
                "radius cell integration requires ChebyshevRadius and shared physical radius"
            )
        if self.normalization != NormalizationSpec(
            kind="continuous_mass", domain="profile_support"
        ):
            raise ValueError("radius cell integration requires continuous profile mass")
        if type(self.base_tile_sites) is not int or self.base_tile_sites < 1:
            raise ValueError("base_tile_sites must be a positive integer")
        if type(self.merge_factor) is not int or self.merge_factor < 2:
            raise ValueError("merge_factor must be an integer at least two")


@dataclass(frozen=True, kw_only=True)
class StatePolicySpec:
    """Initialization or coordinate-update semantics; no execution settings.

    The identifier and immutable settings describe the coordinate policy.
    A backend must recognize the complete contract before executing it.
    """

    id: str
    settings: tuple[tuple[str, float | str | bool], ...] = ()

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("policy ID must be a nonempty string")
        if not isinstance(self.settings, tuple):
            raise TypeError("policy settings must be an immutable tuple")
        names = set()
        for setting in self.settings:
            if not isinstance(setting, tuple) or len(setting) != 2:
                raise TypeError("each setting must be a name/value tuple")
            name, value = setting
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("policy setting names must be unique and nonempty")
            if type(value) not in (float, int, str, bool) or (
                type(value) in (float, int) and not math.isfinite(value)
            ):
                raise ValueError("policy settings must be finite scalar declarations")
            names.add(name)


@dataclass(frozen=True, kw_only=True)
class KernelSpec:
    """An atom function and its coordinate/derivative contract.

    This declaration does not register an implementation or authorize a
    dispatch choice. Geometry/chart are supplied by the enclosing operator.
    Parameter tensors, chunk sizes, checkpointing and CUDA tuning are absent.
    """

    composition: Literal["radial", "separable", "amplitude"]
    profiles: tuple[ProfileBinding, ...] = ()
    parameterization: ParameterizationSpec | None = None
    inner: "KernelSpec | None" = None
    initialization: StatePolicySpec = field(
        default_factory=lambda: StatePolicySpec(id="profile_centers")
    )
    update: StatePolicySpec = field(
        default_factory=lambda: StatePolicySpec(id="geometry")
    )
    revision: int = 1

    def __post_init__(self):
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("kernel revision must be a positive integer")
        if not isinstance(self.profiles, tuple) or not all(
            isinstance(p, ProfileBinding) for p in self.profiles
        ):
            raise TypeError("profiles must be an immutable tuple of bindings")
        from .profiles.fourier import ChebyshevRadiusSpec

        radius_bindings = tuple(
            type(p) is RadiusIntegratedProfileBinding for p in self.profiles
        )
        radius_coordinates = type(self.parameterization) in (
            PhysicalAmpRadiusSpec,
            PolarAmpRadiusSpec,
        )
        radius_profile = any(
            isinstance(p.profile, ChebyshevRadiusSpec) for p in self.profiles
        )
        if (
            radius_coordinates
            or any(radius_bindings)
            or radius_profile
            or self.revision == 7
        ) and (
            not radius_coordinates
            or self.composition != "separable"
            or self.revision != 7
            or len(radius_bindings) != 2
            or not all(radius_bindings)
        ):
            raise ValueError(
                "radius cell masses require revision 7, two radius-integrated factors "
                "and physical or Polar radius coordinates"
            )
        if self.parameterization is not None and not isinstance(
            self.parameterization, ParameterizationSpec
        ):
            raise TypeError("invalid kernel parameterization")
        if not isinstance(self.initialization, StatePolicySpec) or not isinstance(
            self.update, StatePolicySpec
        ):
            raise TypeError("invalid state policy")
        if self.composition == "amplitude":
            if (
                not isinstance(self.inner, KernelSpec)
                or self.profiles
                or self.parameterization is not None
            ):
                raise ValueError("amplitude composition needs only an inner kernel")
        elif self.composition in ("radial", "separable"):
            count = 1 if self.composition == "radial" else 2
            if self.inner is not None or len(self.profiles) != count:
                raise ValueError("profile count differs from kernel composition")
            if self.parameterization is None and any(
                p.parameterization is None for p in self.profiles
            ):
                raise ValueError("each profile needs a bandwidth declaration")
            if self.parameterization is not None and any(
                p.parameterization is not None for p in self.profiles
            ):
                raise ValueError(
                    "bandwidth must be declared once, by the kernel parameterization"
                )
            if self.composition == "separable" and any(
                p.normalization.domain == "operator_sites" for p in self.profiles
            ):
                raise ValueError("separable profiles normalize over chart_sites")
        else:
            raise ValueError("unknown kernel composition")
