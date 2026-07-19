"""ユーザー拡張点であるPolicyの最小契約。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..engine.plan import MutationPlan
from .binding import PolicyBinding
from .schedule import UpdateRequest, UpdateSchedule


class Policy(ABC):
    """prepare時に依存をbindし、step時にMutationPlanを返す。"""

    schedule: UpdateSchedule

    @abstractmethod
    def prepare(self, binding: PolicyBinding) -> None: ...

    @abstractmethod
    def step(self, update: UpdateRequest) -> MutationPlan: ...

    def close(self) -> None:
        """prepare時に登録したhookを解除する。"""
        return None
