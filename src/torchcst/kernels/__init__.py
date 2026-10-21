"""Pure kernel declarations and compositional presets."""

from . import presets
from .fourier_period import HierarchicalFourierPeriodSpec, IntegerFourierPeriodSpec
from .normalization import NormalizationSpec
from .options import KernelOptions
from .parameterizations import (
    AmpWidthSpec,
    BandwidthBounds,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    LogWidthSpec,
    ParameterizationSpec,
    PolarAmpWidthSpec,
)
from .profiles import (
    BiweightSpec,
    DyadicFourierSpec,
    FourierPeakSpec,
    GaussianSpec,
    ProfileSpec,
    QuadraticBSplineSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)
from .profiles.fourier import ChebyshevSquareSpec, DPSSSquareSpec, KaiserSquareSpec
from .profiles.matern import MaternMixtureSpec
from .spec import (
    AtomInit,
    FourierProductKernelSpec,
    KernelSpec,
    ProfileBinding,
    StatePolicySpec,
)
from .support import FourierTileSupportSpec

__all__ = [
    "AmpWidthSpec",
    "AtomInit",
    "BandwidthBounds",
    "BiweightSpec",
    "ChebyshevSquareSpec",
    "DPSSSquareSpec",
    "DirectAmpWidthSpec",
    "DyadicFourierSpec",
    "FixedWidthSpec",
    "FourierPeakSpec",
    "FourierProductKernelSpec",
    "FourierTileSupportSpec",
    "GaussianSpec",
    "HierarchicalFourierPeriodSpec",
    "IntegerFourierPeriodSpec",
    "KaiserSquareSpec",
    "KernelOptions",
    "KernelSpec",
    "LogWidthSpec",
    "MaternMixtureSpec",
    "NormalizationSpec",
    "ParameterizationSpec",
    "PolarAmpWidthSpec",
    "ProfileBinding",
    "ProfileSpec",
    "QuadraticBSplineSpec",
    "StatePolicySpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
    "presets",
]
