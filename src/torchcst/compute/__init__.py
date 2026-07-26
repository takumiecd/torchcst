"""CST compute modules, control families, and backward-capture primitives."""

from .baselines import EntryLinear, NeuronGatedLinear, RankOneLinear
from .capture import (
    BackwardContext,
    CaptureBatch,
    CaptureMode,
    Observation,
    ObservationTiming,
    ReducedObservation,
)
from .batched_conv import batched_conv_dense_weights
from .cst_conv import CSTConv2d, conv2d_isotropic_scales, conv2d_neuron_coordinates
from .cst_linear import CSTLinear
from .graphed_step import GraphedCellRunner, capturable_sgd_step, make_momentum_buffers

ComputeLinear = CSTConv2d | CSTLinear | EntryLinear | NeuronGatedLinear | RankOneLinear

__all__ = [
    "batched_conv_dense_weights",
    "BackwardContext",
    "capturable_sgd_step",
    "CaptureBatch",
    "CaptureMode",
    "ComputeLinear",
    "CSTConv2d",
    "CSTLinear",
    "conv2d_isotropic_scales",
    "conv2d_neuron_coordinates",
    "EntryLinear",
    "GraphedCellRunner",
    "make_momentum_buffers",
    "NeuronGatedLinear",
    "Observation",
    "ObservationTiming",
    "RankOneLinear",
    "ReducedObservation",
]
