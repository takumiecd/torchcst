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
    PhysicalAmpRadiusSpec,
    PhysicalAmpWidthSpec,
    PolarAmpRadiusSpec,
    PolarAmpWidthSpec,
    RadiusBounds,
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
from .profiles.fourier import (
    ChebyshevRadiusSpec,
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
)
from .profiles.matern import MaternMixtureSpec
from .spec import (
    AtomInit,
    CellIntegratedProfileBinding,
    FourierProductKernelSpec,
    KernelSpec,
    ProfileBinding,
    RadiusIntegratedProfileBinding,
    StatePolicySpec,
)
from .support import FourierTileSupportSpec

__all__ = [
    "AmpWidthSpec",
    "AtomInit",
    "BandwidthBounds",
    "BiweightSpec",
    "CellIntegratedProfileBinding",
    "ChebyshevRadiusSpec",
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
    "PhysicalAmpRadiusSpec",
    "PhysicalAmpWidthSpec",
    "PolarAmpRadiusSpec",
    "PolarAmpWidthSpec",
    "ProfileBinding",
    "ProfileSpec",
    "QuadraticBSplineSpec",
    "RadiusBounds",
    "RadiusIntegratedProfileBinding",
    "StatePolicySpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
    "presets",
]
