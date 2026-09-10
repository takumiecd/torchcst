"""Exact visible-space Adam moments queried through atom-local tangents."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchcst._derivatives import AffinePullback

from ..atom_grad import AtomGradRequest
from .atom_rms import AtomBlockMetric
from .base import (
    ExpandedFirstMoment,
    ExpandedSecondMoment,
    FirstMomentComponent,
    MomentContext,
    SecondMomentComponent,
)


def _validate_beta(beta: float) -> float:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    return float(beta)


def _validate_step(beta: float, beta_power: float, next_step: int) -> None:
    if next_step < 1:
        raise ValueError("next_step must be positive")
    if not math.isclose(beta_power, beta ** (next_step - 1)):
        raise ValueError("dense visible moment step or beta mismatch")


@dataclass(frozen=True)
class DenseVisibleFirstMomentState:
    """Visible first-moment EMA with the represented weight shape."""

    value: torch.Tensor
    beta_power: float


class DenseVisibleFirstMoment(FirstMomentComponent):
    """Keep the Adam first moment in the fixed visible weight coordinates."""

    def __init__(self, beta: float):
        self.beta = _validate_beta(beta)

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(visible_gradient=True)

    def initialize(self, context: MomentContext) -> DenseVisibleFirstMomentState:
        return DenseVisibleFirstMomentState(
            value=context.current_point.new_zeros(context.geometry.visible_shape),
            beta_power=1.0,
        )

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, DenseVisibleFirstMomentState):
            raise TypeError("expected a dense visible first-moment state")
        _validate_step(self.beta, state.beta_power, next_step)
        observation.require(self.observation_request)
        assert observation.visible_gradient is not None
        gradient = observation.visible_gradient
        if gradient.shape != context.geometry.visible_shape:
            raise ValueError("visible gradient has the wrong shape")

        value = self.beta * state.value + (1.0 - self.beta) * gradient
        beta_power = state.beta_power * self.beta
        raw_constant = context.geometry.pullback(value, point=context.current_point)
        constant = raw_constant / (1.0 - beta_power)
        zeros = context.current_point.new_zeros(
            *context.current_point.shape, context.current_point.shape[-1]
        )
        return ExpandedFirstMoment(
            raw=AffinePullback(raw_constant, zeros),
            corrected=AffinePullback(constant, zeros),
            pending_state=DenseVisibleFirstMomentState(
                value=value.detach().clone(), beta_power=beta_power
            ),
        )

    def compress(self, expanded, accepted_displacement, context):
        del accepted_displacement, context
        pending = expanded.pending_state
        if not isinstance(pending, DenseVisibleFirstMomentState):
            raise TypeError("expanded first moment belongs to another component")
        return DenseVisibleFirstMomentState(
            value=pending.value.clone(), beta_power=pending.beta_power
        )


@dataclass(frozen=True)
class DenseVisibleSecondMomentState:
    """Visible elementwise second-moment EMA with the weight shape."""

    value: torch.Tensor
    beta_power: float


class DenseVisibleSecondMoment(SecondMomentComponent):
    """Keep dense ``v`` and form exact atom blocks of Adam's denominator."""

    def __init__(
        self,
        beta: float,
        *,
        eps: float,
        row_chunk: int,
        atom_chunk: int,
    ):
        self.beta = _validate_beta(beta)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        for name, value in (("row_chunk", row_chunk), ("atom_chunk", atom_chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.eps = float(eps)
        self.row_chunk = row_chunk
        self.atom_chunk = atom_chunk

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(visible_gradient=True)

    def initialize(self, context: MomentContext) -> DenseVisibleSecondMomentState:
        return DenseVisibleSecondMomentState(
            value=context.current_point.new_zeros(context.geometry.visible_shape),
            beta_power=1.0,
        )

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, DenseVisibleSecondMomentState):
            raise TypeError("expected a dense visible second-moment state")
        _validate_step(self.beta, state.beta_power, next_step)
        observation.require(self.observation_request)
        assert observation.visible_gradient is not None
        gradient = observation.visible_gradient
        if gradient.shape != context.geometry.visible_shape:
            raise ValueError("visible gradient has the wrong shape")

        value = self.beta * state.value + (1.0 - self.beta) * gradient.square()
        beta_power = state.beta_power * self.beta
        corrected_value = value / (1.0 - beta_power)
        denominator = corrected_value.sqrt().add(self.eps)
        geometry = context.geometry
        if not hasattr(geometry, "local_diagonal_gram"):
            raise TypeError("dense visible second moment requires tangent geometry")
        blocks = geometry.local_diagonal_gram(
            context.current_point,
            denominator,
            row_chunk=self.row_chunk,
            atom_chunk=self.atom_chunk,
        )
        blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
        return ExpandedSecondMoment(
            metric=AtomBlockMetric(blocks.detach(), geometry.visible_shape),
            pending_state=DenseVisibleSecondMomentState(
                value=value.detach().clone(), beta_power=beta_power
            ),
        )

    def compress(self, expanded, accepted_displacement, context):
        del accepted_displacement, context
        pending = expanded.pending_state
        if not isinstance(pending, DenseVisibleSecondMomentState):
            raise TypeError("expanded second moment belongs to another component")
        return DenseVisibleSecondMomentState(
            value=pending.value.clone(), beta_power=pending.beta_power
        )
