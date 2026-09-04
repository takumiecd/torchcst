"""Common contract for CST kernel families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Kernel(nn.Module, ABC):
    """Base class for pairwise kernels.

    Kernel-specific derivative contractions will join this contract with the
    derivative milestone. The representation milestone deliberately fixes only
    value evaluation and dimensional compatibility.
    """

    def supports_dimension(self, dimension: int) -> bool:
        """Return whether this kernel can evaluate coordinates of ``dimension``."""

        return dimension > 0

    @abstractmethod
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Evaluate all pairs from ``left[..., N, D]`` and ``right[..., M, D]``."""
