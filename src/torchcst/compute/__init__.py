"""CST compute modules, control families, and backward-capture primitives."""

from .capture import BackwardContext, Observation
from .cst_conv import CSTConv2d, conv2d_neuron_coordinates
from .cst_linear import CSTLinear
from .entry_linear import EntryLinear
from .neuron_gated_linear import NeuronGatedLinear
from .rank_one_linear import RankOneLinear

ComputeLinear = CSTConv2d | CSTLinear | EntryLinear | NeuronGatedLinear | RankOneLinear

__all__ = [
    "BackwardContext",
    "ComputeLinear",
    "CSTConv2d",
    "CSTLinear",
    "conv2d_neuron_coordinates",
    "EntryLinear",
    "NeuronGatedLinear",
    "Observation",
    "RankOneLinear",
]
