"""Declarative scalar profile shapes; no Torch or GPU implementation imports."""

from .base import ProfileSpec
from .compact import BiweightSpec, TriangleSpec, TriweightSpec, WendlandC2Spec
from .gaussian import GaussianSpec

__all__ = [
    "BiweightSpec",
    "GaussianSpec",
    "ProfileSpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
]
