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
from .cst_conv import CSTConv2d, conv2d_neuron_coordinates
from .cst_linear import CSTLinear

ComputeLinear = CSTConv2d | CSTLinear | EntryLinear | NeuronGatedLinear | RankOneLinear

__all__ = [
    "BackwardContext",
    "CaptureBatch",
    "CaptureMode",
    "ComputeLinear",
    "CSTConv2d",
    "CSTLinear",
    "conv2d_neuron_coordinates",
    "EntryLinear",
    "NeuronGatedLinear",
    "Observation",
    "ObservationTiming",
    "RankOneLinear",
    "ReducedObservation",
]
