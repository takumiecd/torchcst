"""Optimizer-side machinery: state following, the coordinate metric, grouping.

Split the way the rest of the package is: ``metric`` is the pure-function world
(tensors in, tensors out, no store and no module), ``precond`` owns the state
and the update rule, ``follower`` keeps slot-indexed optimizer state aligned
with a mutating store, and ``groups`` decides which knobs may share a rate.
"""

from .chart import ChartPullbackAdam, ChartRepulsion
from .follower import OptimizerStateFollower
from .forces import PairRepulsion, SmoothRent
from .groups import parameter_groups
from .precond import CoordPreconditioner
from .pullback import PullbackAdam

__all__ = [
    "ChartPullbackAdam",
    "ChartRepulsion",
    "CoordPreconditioner",
    "OptimizerStateFollower",
    "PairRepulsion",
    "PullbackAdam",
    "SmoothRent",
    "parameter_groups",
]
