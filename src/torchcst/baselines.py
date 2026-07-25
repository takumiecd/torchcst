"""Deprecated control-family namespace: baselines, not CST.

``torchcst.baselines`` exposes the entry (DST/SET/RigL) and chart-free
rank-one control families kept as baseline experiment arms. They are not
CST — use :class:`torchcst.compute.CSTLinear` / :class:`torchcst.compute.
CSTConv2d` for the actual representation. Importing this module (or
``torchcst.compute.baselines``, which backs it) raises a
``DeprecationWarning``.
"""

from __future__ import annotations

from .compute.baselines import EntryLinear, NeuronGatedLinear, RankOneLinear

__all__ = [
    "EntryLinear",
    "NeuronGatedLinear",
    "RankOneLinear",
]
