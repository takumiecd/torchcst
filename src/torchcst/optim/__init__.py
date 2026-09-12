"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .atom_grad import (
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
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
    ExpandedFirstMoment,
    ExpandedMoments,
    ExpandedSecondMoment,
    FirstMomentComponent,
    MomentContext,
    MomentSystem,
    MomentSystemState,
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
    "DeviceBFGS",
    "DeviceRay",
    "ExpandedFirstMoment",
    "ExpandedMoments",
    "ExpandedSecondMoment",
    "FirstMomentComponent",
    "FirstOrderAdamConfig",
    "FullQuartic",
    "FunctionalAdamW",
    "ImplicitLinearAtomGrad",
    "LocalAdamConfig",
    "LocalVisibleAdamConfig",
    "MomentContext",
    "MomentSystem",
    "MomentSystemState",
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
