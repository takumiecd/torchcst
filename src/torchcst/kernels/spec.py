"""Immutable mathematical declarations, independent of backend selection."""

import math
from dataclasses import dataclass, field
from typing import Literal

from .normalization import NormalizationSpec
from .parameterizations import FixedWidthSpec, ParameterizationSpec
from .profiles import ProfileSpec


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


@dataclass(frozen=True, kw_only=True)
class StatePolicySpec:
    """Initialization or coordinate-update semantics; no execution settings.

    The immutable scalar settings retain existing policy options while their
    dedicated initializer/update interfaces are migrated in a later stage.
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
        else:
            raise ValueError("unknown kernel composition")
