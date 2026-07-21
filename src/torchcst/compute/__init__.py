"""CST compute modules, control families, and backward-capture primitives."""

from .capture import BackwardContext, Observation
from .cst_linear import CSTLinear
from .entry_linear import EntryLinear
from .neuron_gated_linear import NeuronGatedLinear
from .rank_one_linear import RankOneLinear

ComputeLinear = CSTLinear | EntryLinear | NeuronGatedLinear | RankOneLinear

__all__ = [
    "BackwardContext",
    "ComputeLinear",
    "CSTLinear",
    "EntryLinear",
    "NeuronGatedLinear",
    "Observation",
    "RankOneLinear",
]
