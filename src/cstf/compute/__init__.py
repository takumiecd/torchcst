"""Sparse compute modules and backward-capture lifecycle primitives."""

from .capture import BackwardContext, Observation
from .entry_linear import EntryLinear
from .rank_one_linear import RankOneLinear

__all__ = ["BackwardContext", "EntryLinear", "Observation", "RankOneLinear"]
