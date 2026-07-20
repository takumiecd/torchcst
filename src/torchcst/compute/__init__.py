"""Sparse compute modules and backward-capture lifecycle primitives."""

from .capture import BackwardContext, Observation
from .cst_linear import CSTLinear
from .entry_linear import EntryLinear
from .rank_one_linear import RankOneLinear

__all__ = ["BackwardContext", "CSTLinear", "EntryLinear", "Observation", "RankOneLinear"]
