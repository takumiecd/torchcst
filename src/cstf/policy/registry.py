"""Policy-owned retirement state, separate from entity IDs and physical slots."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class RetiredCandidateRegistry:
    """Track retired lineage keys per site for the duration of one run."""

    def __init__(self) -> None:
        self._keys: set[tuple[str, int]] = set()

    @staticmethod
    def _values(lineages: torch.Tensor | Iterable[int] | int) -> tuple[int, ...]:
        if isinstance(lineages, torch.Tensor):
            if lineages.ndim != 1 or lineages.dtype != torch.int64:
                raise TypeError("lineages must be a rank-1 int64 tensor")
            return tuple(int(value) for value in lineages.detach().cpu().tolist())
        if isinstance(lineages, bool):
            raise TypeError("lineage must be an int")
        if isinstance(lineages, int):
            return (lineages,)
        return tuple(int(value) for value in lineages)

    def retire(
        self,
        site: str,
        lineages: torch.Tensor | Iterable[int] | int,
    ) -> None:
        for lineage in self._values(lineages):
            self._keys.add((site, lineage))

    def is_retired(self, site: str, lineage: int) -> bool:
        return (site, int(lineage)) in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def snapshot(self) -> frozenset[tuple[str, int]]:
        return frozenset(self._keys)

