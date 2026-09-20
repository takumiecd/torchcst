"""Shared context for CST moment components."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from torchcst._derivatives import FrameGeometry


@dataclass(frozen=True)
class MomentContext:
    """Current site geometry and detached atom point."""

    geometry: FrameGeometry
    current_point: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FrameGeometry):
            raise TypeError("geometry must be a FrameGeometry")
        validated = self.geometry.frame(self.current_point).point
        object.__setattr__(self, "current_point", validated)
