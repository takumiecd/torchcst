"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .atom_grad import (
    AtomGradientObservation,
    AtomGradRequest,
    ImplicitLinearAtomGrad,
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
from .problem import QuarticProblem
from .solvers import FullQuartic, QuarticSolver, QuarticSolveResult

__all__ = [
    "AcceptedFrameFirstMoment",
    "AcceptedFrameFirstMomentState",
    "AtomGradRequest",
    "AtomGradientObservation",
    "ExpandedFirstMoment",
    "ExpandedMoments",
    "ExpandedSecondMoment",
    "FirstMomentComponent",
    "FullQuartic",
    "ImplicitLinearAtomGrad",
    "MomentContext",
    "MomentSystem",
    "MomentSystemState",
    "QuarticProblem",
    "QuarticSolveResult",
    "QuarticSolver",
    "SecondMomentComponent",
    "SeparableDiagonalMetric",
    "SeparableDiagonalSecondMoment",
    "SeparableSecondMomentState",
    "VisibleMetric",
]
