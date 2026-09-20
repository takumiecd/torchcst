"""Functional dense AdamW proposals for coordinated optimizer steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import AdamWConfig


@dataclass(frozen=True)
class DenseAdamWState:
    """Persistent state for one ordinary dense parameter."""

    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor


@dataclass(frozen=True)
class DenseAdamWProposal:
    """A parameter displacement and its uncommitted next state."""

    displacement: Tensor
    pending_state: DenseAdamWState


class FunctionalAdamW:
    """Build AdamW proposals without mutating parameters or persistent state."""

    def __init__(self, config: AdamWConfig) -> None:
        if not isinstance(config, AdamWConfig):
            raise TypeError("config must be an AdamWConfig")
        self.config = config

    def initialize(self, parameter: nn.Parameter) -> DenseAdamWState:
        self._validate_parameter(parameter)
        return DenseAdamWState(
            step=0,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )

    def expand(
        self,
        parameter: nn.Parameter,
        state: DenseAdamWState,
    ) -> DenseAdamWProposal:
        self._validate_parameter(parameter)
        self._validate_state(parameter, state)
        gradient = parameter.grad
        if gradient is None:
            return DenseAdamWProposal(
                displacement=torch.zeros_like(parameter),
                pending_state=state,
            )
        if gradient.is_sparse:
            raise RuntimeError("functional AdamW does not support sparse gradients")
        if gradient.shape != parameter.shape:
            raise ValueError("dense gradient shape does not match its parameter")

        beta1, beta2 = self.config.betas
        detached_gradient = gradient.detach()
        exp_avg = beta1 * state.exp_avg + (1.0 - beta1) * detached_gradient
        exp_avg_sq = (
            beta2 * state.exp_avg_sq
            + (1.0 - beta2) * detached_gradient.square()
        )
        step = state.step + 1
        corrected_avg = exp_avg / (1.0 - beta1**step)
        corrected_avg_sq = exp_avg_sq / (1.0 - beta2**step)
        update = corrected_avg / (corrected_avg_sq.sqrt() + self.config.eps)
        displacement = -self.config.lr * (
            update + self.config.weight_decay * parameter.detach()
        )
        return DenseAdamWProposal(
            displacement=displacement.detach().clone(),
            pending_state=DenseAdamWState(
                step=step,
                exp_avg=exp_avg.detach().clone(),
                exp_avg_sq=exp_avg_sq.detach().clone(),
            ),
        )

    @staticmethod
    def _validate_parameter(parameter: nn.Parameter) -> None:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("parameter must be a torch.nn.Parameter")
        if not parameter.is_floating_point():
            raise TypeError("dense AdamW parameters must be floating point")

    @staticmethod
    def _validate_state(
        parameter: nn.Parameter,
        state: DenseAdamWState,
    ) -> None:
        if not isinstance(state, DenseAdamWState):
            raise TypeError("state must be a DenseAdamWState")
        if state.step < 0:
            raise ValueError("dense AdamW step must be nonnegative")
        for value in (state.exp_avg, state.exp_avg_sq):
            if value.shape != parameter.shape:
                raise ValueError("dense AdamW state shape does not match parameter")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError("dense AdamW state must match parameter device and dtype")
