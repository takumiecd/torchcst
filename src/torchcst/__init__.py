"""Fixed-shape continuous operators for PyTorch."""

from .atoms import Atoms
from .geometry import Chart
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
from .nn import CSTLinear, CSTModule
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
    "Gaussian",
    "Kernel",
    "PolarAmpWidth",
    "NormalizedOptimizerConfig",
    "Profile",
    "QuadraticOptimizerConfig",
    "Separable",
    "Triweight",
    "WendlandC2",
]

from torchcst.optim.parameter import CSTParameterAdam, ParameterAdamConfig

__all__ += ["CSTParameterAdam", "ParameterAdamConfig"]
