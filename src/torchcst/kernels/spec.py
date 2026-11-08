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
    RadiusEnvelopeSpec,
)
from .profiles import ProfileSpec

AtomInit = Literal["balanced", "uniform"]


@dataclass(frozen=True, kw_only=True)
class ProfileBinding:
    """A sampled profile shape, default bandwidth and explicit normalization."""

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


@dataclass(frozen=True, kw_only=True)
class RadiusIntegratedProfileBinding(ProfileBinding):
    """Whole-cell masses with physical support radius and integer hierarchy.

    The density is divided by its total mass, so no extra normalization is
    declared. Local full periods are P=(ceil(2R/T)+1)*T. Base tiles and merge
    factor change the mathematical approximation and belong to this declaration.
    """

    base_tile_sites: int = 1
    merge_factor: int = 2

    def __post_init__(self):
        from .profiles.fourier import ChebyshevSpec

        if type(self.profile) is not ChebyshevSpec or self.parameterization is not None:
            raise ValueError(
                "radius cell integration requires a Chebyshev profile and shared physical radius"
            )
        if self.normalization != NormalizationSpec():
            raise ValueError("cell mass is normalized by definition")
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
class CoordinateSpec:
    """Atom coordinate, initialization and update rules, without a profile.

    Composition supplies the factor(s); this declaration owns dynamic widths.
    It contains no Tensor state or backend execution settings.
    """

    parameterization: ParameterizationSpec
    initialization: StatePolicySpec
    update: StatePolicySpec

    def __post_init__(self):
        if not isinstance(self.parameterization, ParameterizationSpec):
            raise TypeError("invalid coordinate parameterization")
        if not isinstance(self.initialization, StatePolicySpec) or not isinstance(
            self.update, StatePolicySpec
        ):
            raise TypeError("invalid coordinate state policy")


@dataclass(frozen=True, kw_only=True)
class KernelSpec:
    """An atom function and its coordinate/derivative contract.

    This declaration does not register an implementation or authorize a
    dispatch choice. Geometry/chart are supplied by the enclosing operator.
    Parameter tensors, chunk sizes, checkpointing and CUDA tuning are absent.
    """

    composition: Literal["radial", "separable", "amplitude"]
    center_order: Literal["charts"] = "charts"
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
        if self.center_order != "charts":
            raise ValueError("atom centers must follow chart tuple order")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("kernel revision must be a positive integer")
        if not isinstance(self.profiles, tuple) or not all(
            isinstance(p, ProfileBinding) for p in self.profiles
        ):
            raise TypeError("profiles must be an immutable tuple of bindings")
        from .profiles.fourier import ChebyshevSpec

        radius_bindings = tuple(
            type(p) is RadiusIntegratedProfileBinding for p in self.profiles
        )
        radius_coordinates = type(self.parameterization) in (
            PhysicalAmpRadiusSpec,
            PolarAmpRadiusSpec,
            RadiusEnvelopeSpec,
        )
        radius_profile = any(
            isinstance(p.profile, ChebyshevSpec) for p in self.profiles
        )
        if (
            radius_coordinates
            or any(radius_bindings)
            or radius_profile
            or self.revision in (7, 8)
        ) and (
            not radius_coordinates
            or self.composition != "separable"
            or self.revision
            != (8 if type(self.parameterization) is RadiusEnvelopeSpec else 7)
            or len(radius_bindings) != 2
            or not all(radius_bindings)
        ):
            raise ValueError(
                "radius cell masses require their coordinate revision, two radius-integrated factors "
                "and physical, Polar or envelope radius coordinates"
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
