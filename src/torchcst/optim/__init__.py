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
from .follower import OptimizerStateFollower
from .forces import PairRepulsion, SmoothRent
from .groups import parameter_groups
from .precond import CoordPreconditioner
from .pullback import PullbackAdam

__all__ = [
    "CSTOptimizer",
    "ChartPullbackAdam",
    "ChartRepulsion",
    "CoordPreconditioner",
    "OptimizerStateFollower",
    "PairRepulsion",
    "PullbackAdam",
    "PullbackConfig",
    "SmoothRent",
    "continuous_sites",
    "is_continuous_site",
    "parameter_groups",
]
