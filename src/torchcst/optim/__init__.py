"""Optimizer-side machinery for CST pullback training.

``CSTPullbackAdam`` is the model-level API: it owns every CST representation
parameter and exposes the complement for a separate dense optimizer. The
single-site optimizers, coordinator, grouping helper, and structural forces
remain lower-level research mechanisms. ``metric`` is the pure tensor world;
``follower`` keeps slot-indexed state aligned with mutable stores.
"""

from .bandwidth import InverseAmplitudeBandwidth
from .chart import ChartPullbackAdam, ChartRepulsion
from .coordinator import (
    CSTOptimizer,
    PullbackConfig,
    continuous_sites,
    is_continuous_site,
)
from .cst_pullback_adam import CSTPullbackAdam
from .follower import OptimizerStateFollower
from .forces import PairRepulsion, SampledKernelCoherence, SmoothRent
from .groups import parameter_groups
from .precond import CoordPreconditioner
from .pullback import PullbackAdam

__all__ = [
    "CSTOptimizer",
    "CSTPullbackAdam",
    "ChartPullbackAdam",
    "ChartRepulsion",
    "CoordPreconditioner",
    "InverseAmplitudeBandwidth",
    "OptimizerStateFollower",
    "PairRepulsion",
    "PullbackAdam",
    "PullbackConfig",
    "SampledKernelCoherence",
    "SmoothRent",
    "continuous_sites",
    "is_continuous_site",
    "parameter_groups",
]
