"""PolicyからEngineへ渡すentity-neutral mutation plan。"""

from __future__ import annotations

from dataclasses import dataclass

from ..storage.base import Op


@dataclass(frozen=True)
class SiteBatch:
    site: str
    ops: tuple[Op, ...]


@dataclass(frozen=True)
class MutationPlan:
    batches: tuple[SiteBatch, ...] = ()

    @classmethod
    def empty(cls) -> "MutationPlan":
        return cls()
