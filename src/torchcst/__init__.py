"""Fixed-shape continuous operators for PyTorch."""

from .atoms import Atoms
from .geometry import Chart, EuclideanGeometry, Geometry, SphereGeometry
from .kernels import (
    Amplitude,
    AmplitudeBandwidthSeparable,
    AmpWidth,
    Gaussian,
    Kernel,
    PolarAmpWidth,
    Profile,
    Separable,
    Triweight,
    WendlandC2,
)
from .nn import CSTConv2d, CSTLinear, CSTModule
from .optim import (
    CSTSGD,
    AdamRConfig,
    AdamWConfig,
    CSTAdamR,
    CSTImplicitAdam,
    CSTMomentum,
    CSTNormalizedAdam,
    CSTNormalizedMomentum,
    CSTNormalizedRMSProp,
    CSTNormalizedSGD,
    CSTQuadraticAdam,
    CSTQuadraticMomentum,
    CSTQuadraticRMSProp,
    CSTQuadraticSGD,
    CSTRMSProp,
    NormalizedOptimizerConfig,
    QuadraticOptimizerConfig,
)

__all__ = [
    "CSTSGD",
    "AdamRConfig",
    "AdamWConfig",
    "AmpWidth",
    "Amplitude",
    "AmplitudeBandwidthSeparable",
    "Atoms",
    "CSTAdamR",
    "CSTConv2d",
    "CSTImplicitAdam",
    "CSTLinear",
    "CSTModule",
    "CSTMomentum",
    "CSTNormalizedAdam",
    "CSTNormalizedMomentum",
    "CSTNormalizedRMSProp",
    "CSTNormalizedSGD",
    "CSTQuadraticAdam",
    "CSTQuadraticMomentum",
    "CSTQuadraticRMSProp",
    "CSTQuadraticSGD",
    "CSTRMSProp",
    "Chart",
    "EuclideanGeometry",
    "Gaussian",
    "Geometry",
    "Kernel",
    "NormalizedOptimizerConfig",
    "PolarAmpWidth",
    "Profile",
    "QuadraticOptimizerConfig",
    "Separable",
    "SphereGeometry",
    "Triweight",
    "WendlandC2",
]

from torchcst.optim.parameter import CSTParameterAdam, ParameterAdamConfig

__all__ += ["CSTParameterAdam", "ParameterAdamConfig"]
