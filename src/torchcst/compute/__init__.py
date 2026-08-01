"""CST compute modules, control families, and backward-capture primitives."""

from .baselines import EntryLinear, NeuronGatedLinear, RankOneLinear
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
from .cst_block import CSTBlock
from .cst_conv import CSTConv2d, conv2d_isotropic_scales, conv2d_neuron_coordinates
from .cst_linear import CSTLinear
from .graphed_step import GraphedCellRunner, capturable_sgd_step, make_momentum_buffers

ComputeLinear = (
    CSTBlock | CSTConv2d | CSTLinear | EntryLinear | NeuronGatedLinear | RankOneLinear
)

__all__ = [
    "BackwardContext",
    "CSTBlock",
    "CSTConv2d",
    "CSTLinear",
    "CaptureBatch",
    "CaptureMode",
    "ComputeLinear",
    "EntryLinear",
    "GraphedCellRunner",
    "NeuronGatedLinear",
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
