"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .atom_grad import (
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
    LinearJGHAtomGrad,
)
from .config import (
    AdamWConfig,
    DenseVisibleAdamConfig,
    FirstOrderAdamConfig,
    LocalAdamConfig,
    LocalVisibleAdamConfig,
    NormalizedOptimizerConfig,
    SecondOrderAdamConfig,
)
from .dense import (
    DenseAdamWProposal,
    DenseAdamWState,
    FunctionalAdamW,
)
from .dense_visible import CSTDenseVisibleAdam
from .first_order import CSTAdam
from .local import CSTLocalAdam
from .local_visible import CSTLocalVisibleAdam
from .moments import (
    AcceptedFrameFirstMoment,
    AcceptedFrameFirstMomentState,
    DenominatorMoment,
    DenominatorMomentState,
    DenominatorPolynomial,
    ExpandedDenominatorMoment,
    ExpandedFirstMoment,
    ExpandedMoments,
    ExpandedNumeratorMoment,
    ExpandedSecondMoment,
    FirstMomentComponent,
    MomentContext,
    MomentSystem,
    MomentSystemState,
    NumeratorMoment,
    NumeratorMomentState,
    SecondMomentComponent,
    SeparableDiagonalMetric,
    SeparableDiagonalSecondMoment,
    SeparableSecondMomentState,
    VisibleMetric,
)
from .normalized_optimizer import (
    CSTSGD,
    CSTImplicitAdam,
    CSTMomentum,
    CSTNormalizedAdam,
    CSTRMSProp,
)
from .optimizer import CSTStepResult
from .problem import QuarticProblem
from .second_order import CSTSecondOrderAdam
from .solvers import (
    BallNewton,
    DeviceBFGS,
    DeviceRay,
    FullQuartic,
    NormalizedBoxFixedPointSolver,
    NormalizedEvaluation,
    NormalizedFixedPointSolver,
    NormalizedSolver,
    NormalizedSolveResult,
    NormalizedUpdateProblem,
    ProjectedLBFGS,
    QuarticSolver,
    QuarticSolveResult,
    SubspaceQuartic,
)

__all__ = [
    "CSTSGD",
    "AcceptedFrameFirstMoment",
    "AcceptedFrameFirstMomentState",
    "AdamWConfig",
    "AtomGradRequest",
    "AtomGradientObservation",
    "BallNewton",
    "CSTAdam",
    "CSTDenseVisibleAdam",
    "CSTImplicitAdam",
    "CSTLocalAdam",
    "CSTLocalVisibleAdam",
    "CSTMomentum",
    "CSTNormalizedAdam",
    "CSTRMSProp",
    "CSTSecondOrderAdam",
    "CSTStepResult",
    "DenominatorMoment",
    "DenominatorMomentState",
    "DenominatorPolynomial",
    "DenseAdamWProposal",
    "DenseAdamWState",
    "DenseVisibleAdamConfig",
    "DeviceBFGS",
    "DeviceRay",
    "ExpandedDenominatorMoment",
    "ExpandedFirstMoment",
    "ExpandedMoments",
    "ExpandedNumeratorMoment",
    "ExpandedSecondMoment",
    "FirstMomentComponent",
    "FirstOrderAdamConfig",
    "FullQuartic",
    "FunctionalAdamW",
    "ImplicitLinearAtomGrad",
    "LinearJGHAtomGrad",
    "LocalAdamConfig",
    "LocalVisibleAdamConfig",
    "MomentContext",
    "MomentSystem",
    "MomentSystemState",
    "NormalizedBoxFixedPointSolver",
    "NormalizedEvaluation",
    "NormalizedFixedPointSolver",
    "NormalizedOptimizerConfig",
    "NormalizedSolveResult",
    "NormalizedSolver",
    "NormalizedUpdateProblem",
    "NumeratorMoment",
    "NumeratorMomentState",
    "ProjectedLBFGS",
    "QuarticProblem",
    "QuarticSolveResult",
    "QuarticSolver",
    "SecondMomentComponent",
    "SecondOrderAdamConfig",
    "SeparableDiagonalMetric",
    "SeparableDiagonalSecondMoment",
    "SeparableSecondMomentState",
    "SubspaceQuartic",
    "VisibleMetric",
]

from torchcst.optim.parameter import CSTParameterAdam, ParameterAdamConfig

__all__ += ["CSTParameterAdam", "ParameterAdamConfig"]
