"""Declarative scalar profile shapes; no Torch or GPU implementation imports."""

from .base import ProfileSpec
from .compact import BiweightSpec, TriangleSpec, TriweightSpec, WendlandC2Spec
from .fourier import (
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
    PeriodicFourierSpec,
)
from .gaussian import GaussianSpec

__all__ = [
    "BiweightSpec",
    "ChebyshevSquareSpec",
    "DPSSSquareSpec",
    "GaussianSpec",
    "KaiserSquareSpec",
    "PeriodicFourierSpec",
    "ProfileSpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
]
