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
    ExpandedFirstMoment,
    ExpandedDenominatorMoment,
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
from .optimizer import CSTStepResult
from .problem import QuarticProblem
from .second_order import CSTSecondOrderAdam
from .solvers import (
    BallNewton,
    DeviceBFGS,
    DeviceRay,
    FullQuartic,
    NormalizedFixedPointSolver,
    NormalizedSolveResult,
    NormalizedSolver,
    NormalizedUpdateProblem,
    ProjectedLBFGS,
    QuarticSolver,
    QuarticSolveResult,
    SubspaceQuartic,
)

__all__ = [
    "AcceptedFrameFirstMoment",
    "AcceptedFrameFirstMomentState",
    "AdamWConfig",
    "AtomGradRequest",
    "AtomGradientObservation",
    "BallNewton",
    "CSTAdam",
    "CSTDenseVisibleAdam",
    "CSTLocalAdam",
    "CSTLocalVisibleAdam",
    "CSTSecondOrderAdam",
    "CSTStepResult",
    "DenseAdamWProposal",
    "DenseAdamWState",
    "DenseVisibleAdamConfig",
    "DenominatorMoment",
    "DenominatorMomentState",
    "DenominatorPolynomial",
    "DeviceBFGS",
    "DeviceRay",
    "ExpandedFirstMoment",
    "ExpandedDenominatorMoment",
    "ExpandedMoments",
    "ExpandedNumeratorMoment",
    "ExpandedSecondMoment",
    "FirstMomentComponent",
    "FirstOrderAdamConfig",
    "FullQuartic",
    "NormalizedFixedPointSolver",
    "NormalizedSolveResult",
    "NormalizedSolver",
    "NormalizedUpdateProblem",
    "FunctionalAdamW",
    "ImplicitLinearAtomGrad",
    "LocalAdamConfig",
    "LocalVisibleAdamConfig",
    "LinearJGHAtomGrad",
    "MomentContext",
    "MomentSystem",
    "MomentSystemState",
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
