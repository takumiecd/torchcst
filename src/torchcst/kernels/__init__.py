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
    PolarAmpWidthSpec,
)
from .profiles import (
    BiweightSpec,
    GaussianSpec,
    ProfileSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)
from .profiles.fourier import (
    ChebyshevSquareSpec,
    DPSSSquareSpec,
    KaiserSquareSpec,
    KaiserTemplateSpec,
)
from .spec import (
    AtomInit,
    FourierProductKernelSpec,
    FourierTwoTileKernelSpec,
    KernelSpec,
    ProfileBinding,
    StatePolicySpec,
)
from .support import FourierTileSupportSpec, FourierTwoTileSupportSpec

__all__ = [
    "AmpWidthSpec",
    "AtomInit",
    "BandwidthBounds",
    "BiweightSpec",
    "ChebyshevSquareSpec",
    "DPSSSquareSpec",
    "DirectAmpWidthSpec",
    "FixedWidthSpec",
    "FourierProductKernelSpec",
    "FourierTileSupportSpec",
    "FourierTwoTileKernelSpec",
    "FourierTwoTileSupportSpec",
    "GaussianSpec",
    "KaiserSquareSpec",
    "KaiserTemplateSpec",
    "KernelOptions",
    "KernelSpec",
    "LogWidthSpec",
    "NormalizationSpec",
    "ParameterizationSpec",
    "PolarAmpWidthSpec",
    "ProfileBinding",
    "ProfileSpec",
    "StatePolicySpec",
    "TriangleSpec",
    "TriweightSpec",
    "WendlandC2Spec",
    "presets",
]
