"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .atom_grad import (
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
)
from .config import AdamWConfig, ImplicitAdamConfig
from .dense import (
    DenseAdamWProposal,
    DenseAdamWState,
    FunctionalAdamW,
)
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
from .optimizer import CSTOptimizer, CSTStepResult
from .problem import QuarticProblem
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
    "CSTOptimizer",
    "CSTStepResult",
    "DenseAdamWProposal",
    "DenseAdamWState",
    "DeviceBFGS",
    "DeviceRay",
    "ExpandedFirstMoment",
    "ExpandedMoments",
    "ExpandedSecondMoment",
    "FirstMomentComponent",
    "FullQuartic",
    "FunctionalAdamW",
    "ImplicitAdamConfig",
    "ImplicitLinearAtomGrad",
    "MomentContext",
    "MomentSystem",
    "MomentSystemState",
    "ProjectedLBFGS",
    "QuarticProblem",
    "QuarticSolveResult",
    "QuarticSolver",
    "SecondMomentComponent",
    "SeparableDiagonalMetric",
    "SeparableDiagonalSecondMoment",
    "SeparableSecondMomentState",
    "SubspaceQuartic",
    "VisibleMetric",
]
