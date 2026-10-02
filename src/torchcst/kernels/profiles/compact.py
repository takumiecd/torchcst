"""Declarations of compact radial function shapes."""

from dataclasses import dataclass, field

from .base import ProfileSpec


@dataclass(frozen=True, kw_only=True)
class TriweightSpec(ProfileSpec):
    """f(q) = max(0, 1-q)^3; q is squared distance / sigma^2."""

    id: str = field(default="triweight", init=False)


@dataclass(frozen=True, kw_only=True)
class BiweightSpec(ProfileSpec):
    """f(q) = max(0, 1-q)^2."""

    id: str = field(default="biweight", init=False)


@dataclass(frozen=True, kw_only=True)
class TriangleSpec(ProfileSpec):
    """f(q) = max(0, 1-sqrt(q))."""

    id: str = field(default="triangle", init=False)


@dataclass(frozen=True, kw_only=True)
class WendlandC2Spec(ProfileSpec):
    """f(q) = max(0, 1-sqrt(q))^4 * (4*sqrt(q)+1)."""

    id: str = field(default="wendland_c2", init=False)
