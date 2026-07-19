"""ユーザー拡張点であるPolicyの最小契約。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..engine.plan import MutationPlan
from .binding import PolicyBinding
from .schedule import UpdateRequest, UpdateSchedule


class SynapsePolicy(Protocol):
    """全体Policy内でsynapse側を担当するコンポーネント。"""

    def prepare(self, binding: PolicyBinding) -> None: ...

    def step(self, update: UpdateRequest) -> MutationPlan: ...


class NeuronPolicy(Protocol):
    """全体Policy内でneuron側を担当するコンポーネント。"""

    def prepare(self, binding: PolicyBinding) -> None: ...

    def step(self, update: UpdateRequest) -> MutationPlan: ...


class Policy(ABC):
    """prepare時に依存をbindし、step時にMutationPlanを返す。"""

    schedule: UpdateSchedule

    synapse: SynapsePolicy | None = None
    neuron: NeuronPolicy | None = None

    @abstractmethod
    def prepare(self, binding: PolicyBinding) -> None: ...

    @abstractmethod
    def step(self, update: UpdateRequest) -> MutationPlan: ...

    def close(self) -> None:
        """prepare時に登録したhookを解除する。"""
        return None
