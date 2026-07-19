"""Backward観測のentity別契約。"""

from __future__ import annotations

from typing import Protocol

from ..backward import ModuleGradRecord


class BackwardObserver(Protocol):
    """backward中にrecordを受け取り、要約状態だけを更新する。"""

    def observe(self, record: ModuleGradRecord) -> None: ...


class SynapseBackwardObserver(BackwardObserver, Protocol):
    """synapse由来のgradient recordを扱うobserver。"""


class NeuronBackwardObserver(BackwardObserver, Protocol):
    """neuron由来のgradient recordを扱うobserver。"""
