"""Exact-loss acceptance policies for optimizer transactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

LossEvaluator = Callable[[float], Tensor]


@dataclass(frozen=True)
class AcceptanceResult:
    """The selected scale and exact loss for one optimizer proposal."""

    accepted: bool
    scale: float
    loss: Tensor
    trials: int


class AcceptancePolicy(ABC):
    """Choose a proposal scale without owning parameters or optimizer state."""

    @abstractmethod
    def select(
        self,
        base_loss: Tensor,
        evaluate: LossEvaluator,
    ) -> AcceptanceResult: ...


class ExactLossAcceptance(AcceptancePolicy):
    """Accept the first finite backtracking candidate that does not raise loss."""

    def __init__(
        self,
        *,
        backtrack_factor: float = 0.5,
        max_trials: int = 8,
    ) -> None:
        if not 0.0 < backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must satisfy 0 < value < 1")
        if isinstance(max_trials, bool) or not isinstance(max_trials, int):
            raise TypeError("max_trials must be an integer")
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        self.backtrack_factor = float(backtrack_factor)
        self.max_trials = max_trials

    def select(
        self,
        base_loss: Tensor,
        evaluate: LossEvaluator,
    ) -> AcceptanceResult:
        base = self._loss(base_loss, name="base loss")
        if not torch.isfinite(base):
            raise FloatingPointError("base loss must be finite")

        scale = 1.0
        for trial in range(1, self.max_trials + 1):
            candidate = self._loss(evaluate(scale), name="candidate loss")
            if torch.isfinite(candidate) and bool(candidate <= base):
                return AcceptanceResult(
                    accepted=True,
                    scale=scale,
                    loss=candidate,
                    trials=trial,
                )
            scale *= self.backtrack_factor

        return AcceptanceResult(
            accepted=False,
            scale=0.0,
            loss=base,
            trials=self.max_trials,
        )

    @staticmethod
    def _loss(value: Tensor, *, name: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return value.detach().reshape(())
