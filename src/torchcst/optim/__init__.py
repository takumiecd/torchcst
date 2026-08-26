"""Optimizer-side machinery: state following, the coordinate metric, grouping.

Split the way the rest of the package is: ``metric`` is the pure-function world
(tensors in, tensors out, no store and no module), ``precond`` and ``pullback``
own the state and the update rules, ``follower`` keeps slot-indexed optimizer
state aligned with a mutating store, ``groups`` decides which knobs may share a
rate, and ``coordinator`` is the one place that decides which optimizer owns
which parameter across a whole multi-site model.
"""

from .chart import ChartPullbackAdam, ChartRepulsion
from .coordinator import (
    CSTOptimizer,
    PullbackConfig,
    continuous_sites,
    is_continuous_site,
)
from .cst_pullback_adam import CSTPullbackAdam
from .follower import OptimizerStateFollower
from .forces import PairRepulsion, SmoothRent
from .groups import parameter_groups
from .precond import CoordPreconditioner
from .pullback import PullbackAdam
from .wbasis import (
    WhitenedAmplitudeBasis,
    amplitude_leaf,
    install_whitened_basis,
    refresh_whitened_basis,
)

__all__ = [
    "CSTOptimizer",
    "CSTPullbackAdam",
    "ChartPullbackAdam",
    "ChartRepulsion",
    "CoordPreconditioner",
    "OptimizerStateFollower",
    "PairRepulsion",
    "PullbackAdam",
    "PullbackConfig",
    "SmoothRent",
    "WhitenedAmplitudeBasis",
    "amplitude_leaf",
    "continuous_sites",
    "install_whitened_basis",
    "is_continuous_site",
    "parameter_groups",
    "refresh_whitened_basis",
]
