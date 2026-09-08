"""Separable visible-diagonal second moments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._runtime.validation import require

from ..atom_grad import AtomGradientObservation, AtomGradRequest
from .base import ExpandedSecondMoment, MomentContext, SecondMomentComponent


@dataclass(frozen=True)
class SeparableSecondMomentState:
    """Raw row and column square EMAs."""

    row: Tensor
    column: Tensor
    beta_power: float


class SeparableDiagonalMetric:
    """A matrix-shaped diagonal metric reconstructed from row/column state."""

    def __init__(self, row: Tensor, column: Tensor, *, eps: float) -> None:
        if row.ndim != 1 or column.ndim != 1:
            raise ValueError("row and column must be vectors")
        if row.device != column.device or row.dtype != column.dtype:
            raise ValueError("row and column must share one device and dtype")
        require(
            (row >= 0).all() & (column >= 0).all(),
            "second-moment values must be nonnegative",
        )
        if eps <= 0:
            raise ValueError("eps must be positive")
        self._row = row.detach().clone()
        self._column = column.detach().clone()
        self.eps = float(eps)
        normalizer = self._row.mean().clamp_min(torch.finfo(self._row.dtype).tiny)
        self._row_weight = (self._row / normalizer).sqrt()
        self._column_weight = self._column.sqrt()

    @property
    def visible_shape(self) -> tuple[int, int]:
        return self._row.shape[0], self._column.shape[0]

    @property
    def row(self) -> Tensor:
        return self._row.clone()

    @property
    def column(self) -> Tensor:
        return self._column.clone()

    def diagonal(self) -> Tensor:
        """Materialize ``sqrt(v_tilde) + eps`` for diagnostics and oracles."""

        return self._row_weight[:, None] * self._column_weight[None, :] + self.eps

    def separable_weights(self) -> tuple[Tensor, Tensor, float]:
        """Return s, t, eps with metric diagonal s[:, None] * t + eps."""

        return self._row_weight.clone(), self._column_weight.clone(), self.eps

    def apply(self, value: Tensor) -> Tensor:
        self._validate_visible(value)
        return (
            self._row_weight[:, None] * value * self._column_weight[None, :]
            + self.eps * value
        )

    def inner(self, left: Tensor, right: Tensor) -> Tensor:
        self._validate_visible(left)
        self._validate_visible(right)
        return (left * self.apply(right)).sum()

    def _validate_visible(self, value: Tensor) -> None:
        if value.shape != self.visible_shape:
            raise ValueError(f"value must have shape {list(self.visible_shape)}")
        if value.device != self._row.device or value.dtype != self._row.dtype:
            raise ValueError("value must match metric device and dtype")


class SeparableDiagonalSecondMoment(SecondMomentComponent):
    """Adam-style row/column square EMAs for a matrix-shaped visible operator."""

    def __init__(self, beta: float, *, eps: float = 1e-8) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.beta = float(beta)
        self.eps = float(eps)

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(row_square=True, column_square=True)

    def initialize(self, context: MomentContext) -> SeparableSecondMomentState:
        if len(context.geometry.visible_shape) != 2:
            raise ValueError("separable second moment requires a matrix-shaped site")
        rows, columns = context.geometry.visible_shape
        return SeparableSecondMomentState(
            row=context.current_point.new_zeros(rows),
            column=context.current_point.new_zeros(columns),
            beta_power=1.0,
        )

    def expand(
        self,
        state: object,
        observation: AtomGradientObservation,
        context: MomentContext,
        *,
        next_step: int,
    ) -> ExpandedSecondMoment:
        if not isinstance(state, SeparableSecondMomentState):
            raise TypeError("state must be a SeparableSecondMomentState")
        if next_step < 1:
            raise ValueError("next_step must be positive")
        expected_beta_power = self.beta ** (next_step - 1)
        if not math.isclose(state.beta_power, expected_beta_power):
            raise ValueError("second-moment state does not match step or beta")
        observation.require(self.observation_request)
        assert observation.row_square is not None
        assert observation.column_square is not None
        if observation.row_square.shape != state.row.shape:
            raise ValueError("row-square observation has the wrong shape")
        if observation.column_square.shape != state.column.shape:
            raise ValueError("column-square observation has the wrong shape")

        row = self.beta * state.row + (1.0 - self.beta) * observation.row_square
        column = (
            self.beta * state.column + (1.0 - self.beta) * observation.column_square
        )
        beta_power = state.beta_power * self.beta
        correction = 1.0 - beta_power
        corrected_row = row / correction
        corrected_column = column / correction
        pending = SeparableSecondMomentState(
            row=row.detach().clone(),
            column=column.detach().clone(),
            beta_power=beta_power,
        )
        return ExpandedSecondMoment(
            metric=SeparableDiagonalMetric(
                corrected_row,
                corrected_column,
                eps=self.eps,
            ),
            pending_state=pending,
        )

    def compress(
        self,
        expanded: ExpandedSecondMoment,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> SeparableSecondMomentState:
        del accepted_displacement, context
        pending = expanded.pending_state
        if not isinstance(pending, SeparableSecondMomentState):
            raise TypeError("expanded second moment belongs to another component")
        return SeparableSecondMomentState(
            row=pending.row.clone(),
            column=pending.column.clone(),
            beta_power=pending.beta_power,
        )
