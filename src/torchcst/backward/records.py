"""PyTorch hookからPolicyへ渡せるdetach済みgradient record。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


class CoordinateGradientProvider(Protocol):
    """連続座標上のsynapse weight gradientを評価する能力。"""

    def weight_gradients(
        self,
        input: Tensor,
        grad_output: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """各(source, target)に対応するdL/dwを返す。"""
        ...


@dataclass(frozen=True)
class LinearGradRecord:
    """CSTLinearのforward factsとoutput gradientを結びつけた1観測。"""

    site: str
    version: int
    input: Tensor
    grad_output: Tensor
    gradients: CoordinateGradientProvider
