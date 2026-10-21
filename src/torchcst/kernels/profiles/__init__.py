"""Declarative scalar profile shapes; no Torch or GPU implementation imports."""

from .base import ProfileSpec
from .bspline import QuadraticBSplineSpec
from .compact import BiweightSpec, TriangleSpec, TriweightSpec, WendlandC2Spec
from .dyadic_fourier import DyadicFourierSpec
from .fourier import (
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
    PeriodicFourierSpec,
)
from .fourier_peak import FourierPeakSpec
from .gaussian import GaussianSpec
from .matern import MaternMixtureSpec

__all__ = [
    "BiweightSpec",
    "ChebyshevSquareSpec",
    "DPSSSquareSpec",
    "DyadicFourierSpec",
    "FourierPeakSpec",
    "GaussianSpec",
    "KaiserSquareSpec",
    "MaternMixtureSpec",
    "PeriodicFourierSpec",
    "ProfileSpec",
    "QuadraticBSplineSpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
]
