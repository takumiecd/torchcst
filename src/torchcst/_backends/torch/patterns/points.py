"""PyTorch evaluation of points coordinate declarations."""

from __future__ import annotations

from torch import Tensor


def positions(self, indices: Tensor) -> Tensor:
    return self.coordinates.index_select(0, indices)


def bounds(self) -> tuple[Tensor, Tensor]:
    return self.coordinates.amin(0), self.coordinates.amax(0)
