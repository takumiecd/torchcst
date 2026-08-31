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
from .p2 import (
    ContractedP2Model,
    CSTP2TrustRegion,
    P2ModelEMA,
    P2StepEMA,
    TrustRegionResult,
    contracted_cst_linear_p2_model,
    contracted_p2_model,
    cst_linear_p2_reducer,
    solve_block_trust_region,
)
from .precond import CoordPreconditioner
from .pullback import PullbackAdam

__all__ = [
    "CSTOptimizer",
    "CSTPullbackAdam",
    "ChartPullbackAdam",
    "ChartRepulsion",
    "CoordPreconditioner",
    "ContractedP2Model",
    "CSTP2TrustRegion",
    "InverseAmplitudeBandwidth",
    "OptimizerStateFollower",
    "PairRepulsion",
    "P2ModelEMA",
    "P2StepEMA",
    "PullbackAdam",
    "PullbackConfig",
    "SampledKernelCoherence",
    "SmoothRent",
    "TrustRegionResult",
    "contracted_cst_linear_p2_model",
    "contracted_p2_model",
    "cst_linear_p2_reducer",
    "continuous_sites",
    "is_continuous_site",
    "parameter_groups",
    "solve_block_trust_region",
]
