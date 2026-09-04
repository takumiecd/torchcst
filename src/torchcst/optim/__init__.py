"""Model-level CST optimization.

The public optimizer is introduced after the derivative operator contract is
validated against dense autograd oracles.
"""

from .acceptance import (
    AcceptancePolicy,
    AcceptanceResult,
    ExactLossAcceptance,
)
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
from .solvers import FullQuartic, QuarticSolver, QuarticSolveResult

__all__ = [
    "AcceptancePolicy",
    "AcceptanceResult",
    "AcceptedFrameFirstMoment",
    "AcceptedFrameFirstMomentState",
    "AdamWConfig",
    "AtomGradRequest",
    "AtomGradientObservation",
    "CSTOptimizer",
    "CSTStepResult",
    "DenseAdamWProposal",
    "DenseAdamWState",
    "ExactLossAcceptance",
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
    "QuarticProblem",
    "QuarticSolveResult",
    "QuarticSolver",
    "SecondMomentComponent",
    "SeparableDiagonalMetric",
    "SeparableDiagonalSecondMoment",
    "SeparableSecondMomentState",
    "VisibleMetric",
]
