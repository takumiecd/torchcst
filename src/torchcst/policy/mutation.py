"""backward観測から分離されたmutation判断の契約。"""

from __future__ import annotations

from typing import Protocol

from ..engine.plan import MutationPlan
from .binding import PolicyBinding
from .schedule import UpdateRequest


class MutationPolicy(Protocol):
    """Storageへ適用するMutationPlanを生成するentity別policy。"""

    def prepare(self, binding: PolicyBinding) -> None: ...

    def step(self, update: UpdateRequest) -> MutationPlan: ...
