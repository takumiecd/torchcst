"""Pure kernel declarations and compositional presets."""

from . import presets
from .normalization import NormalizationSpec
from .options import KernelOptions
from .parameterizations import (
    AmpWidthSpec,
    BandwidthBounds,
    DirectAmpWidthSpec,
    FixedWidthSpec,
    LogWidthSpec,
    ParameterizationSpec,
    PhysicalAmpWidthSpec,
    PolarAmpWidthSpec,
)
from .profiles import (
    BiweightSpec,
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
    CellIntegratedProfileBinding,
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
    "CellIntegratedProfileBinding",
    "ChebyshevSquareSpec",
    "DPSSSquareSpec",
    "DirectAmpWidthSpec",
    "FixedWidthSpec",
    "FourierProductKernelSpec",
    "FourierTileSupportSpec",
    "GaussianSpec",
    "KaiserSquareSpec",
    "KernelOptions",
    "KernelSpec",
    "LogWidthSpec",
    "MaternMixtureSpec",
    "NormalizationSpec",
    "ParameterizationSpec",
    "PhysicalAmpWidthSpec",
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
