"""CST compute modules, control families, and backward-capture primitives."""

from .baselines import EntryLinear, NeuronGatedLinear, RankOneLinear
from .boundary import CSTBoundary
from .capture import (
    BackwardContext,
    CaptureBatch,
    CaptureMode,
    Observation,
    ObservationTiming,
    ReducedObservation,
    flatten_capture_pair,
    register_capture_hook,
)
from .batched_conv import batched_conv_dense_weights
from .cst_conv import CSTConv2d, conv2d_isotropic_scales, conv2d_neuron_coordinates
from .cst_linear import CSTLinear
from .depthwise_conv import DepthwiseCSTConv2d
from .offset_conv import OffsetCSTConv2d
from .graphed_step import GraphedCellRunner, capturable_sgd_step, make_momentum_buffers

# CSTBoundary is not a site module: the engine never sees one, only the maps
# whose output boundary it owns (reachable via map.out_boundary).
ComputeLinear = (
    CSTConv2d
    | CSTLinear
    | EntryLinear
    | NeuronGatedLinear
    | OffsetCSTConv2d
    | RankOneLinear
)

__all__ = [
    "BackwardContext",
    "CSTBoundary",
    "CSTConv2d",
    "CSTLinear",
    "DepthwiseCSTConv2d",
    "CaptureBatch",
    "CaptureMode",
    "ComputeLinear",
    "EntryLinear",
    "GraphedCellRunner",
    "NeuronGatedLinear",
    "OffsetCSTConv2d",
    "Observation",
    "ObservationTiming",
    "RankOneLinear",
    "ReducedObservation",
    "batched_conv_dense_weights",
    "capturable_sgd_step",
    "conv2d_isotropic_scales",
    "conv2d_neuron_coordinates",
    "flatten_capture_pair",
    "make_momentum_buffers",
    "register_capture_hook",
]
