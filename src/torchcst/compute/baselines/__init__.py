"""Control-family compute modules: baselines, not CST.

``EntryLinear`` and ``RankOneLinear`` are optional control families kept for
the DST/SET/RigL and chart-free rank-one baseline arms of the experiment
suite. They are deprecated in favor of :class:`torchcst.compute.CSTLinear`
and :class:`torchcst.compute.CSTConv2d` and must not be treated as the main
representation. They are intentionally not deleted: removing them would force
the baseline arms out of this codebase.
"""

from __future__ import annotations

import warnings

from .entry_linear import EntryLinear
from .neuron_gated_linear import NeuronGatedLinear
from .rank_one_linear import RankOneLinear

warnings.warn(
    "torchcst.compute.baselines (EntryLinear, RankOneLinear, "
    "NeuronGatedLinear) are deprecated control families, not CST. Use "
    "torchcst.compute.CSTLinear / torchcst.compute.CSTConv2d instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EntryLinear",
    "NeuronGatedLinear",
    "RankOneLinear",
]
