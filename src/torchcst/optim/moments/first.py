"""Accepted-representation-frame first moments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._derivatives import AffinePullback, RepresentationFrame

from ..atom_grad import AtomGradientObservation, AtomGradRequest
from .base import ExpandedFirstMoment, FirstMomentComponent, MomentContext


@dataclass(frozen=True)
class AcceptedFrameFirstMomentState:
    """Raw EMA coefficients represented in the last accepted frame."""

    alpha: Tensor
    frame: RepresentationFrame
    beta_power: float


@dataclass(frozen=True)
class _AcceptedFramePendingState:
    beta_power: float


class AcceptedFrameFirstMoment(FirstMomentComponent):
    """Compact first-moment EMA whose coordinates follow accepted frames."""

    def __init__(
        self,
        beta: float,
        *,
        damping: float = 0.0,
        rtol: float | None = None,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if damping < 0:
            raise ValueError("damping must be nonnegative")
        if rtol is not None and rtol < 0:
            raise ValueError("rtol must be nonnegative")
        self.beta = float(beta)
        self.damping = float(damping)
        self.rtol = rtol

    @property
    def observation_request(self) -> AtomGradRequest:
        return AtomGradRequest(jg=True, gh=True)

    def initialize(self, context: MomentContext) -> AcceptedFrameFirstMomentState:
        frame = context.geometry.frame(context.current_point)
        return AcceptedFrameFirstMomentState(
            alpha=torch.zeros_like(context.current_point),
            frame=frame,
            beta_power=1.0,
        )

    def expand(
        self,
        state: object,
        observation: AtomGradientObservation,
        context: MomentContext,
        *,
        next_step: int,
    ) -> ExpandedFirstMoment:
        if not isinstance(state, AcceptedFrameFirstMomentState):
            raise TypeError("state must be an AcceptedFrameFirstMomentState")
        if next_step < 1:
            raise ValueError("next_step must be positive")
        expected_beta_power = self.beta ** (next_step - 1)
        if not math.isclose(state.beta_power, expected_beta_power):
            raise ValueError("first-moment state does not match step or beta")
        observation.require(self.observation_request)
        assert observation.jg is not None
        assert observation.gh is not None

        current = AffinePullback(observation.jg, observation.gh)
        if next_step == 1:
            previous = AffinePullback(
                torch.zeros_like(observation.jg),
                torch.zeros_like(observation.gh),
            )
        else:
            previous = context.geometry.pullback_from_frame(
                current_point=context.current_point,
                source_frame=state.frame,
                source_coefficients=state.alpha,
            )

        raw = previous.scaled(self.beta) + current.scaled(1.0 - self.beta)
        beta_power = state.beta_power * self.beta
        correction = 1.0 - beta_power
        corrected = raw.scaled(1.0 / correction)
        return ExpandedFirstMoment(
            raw=raw,
            corrected=corrected,
            pending_state=_AcceptedFramePendingState(beta_power=beta_power),
        )

    def compress(
        self,
        expanded: ExpandedFirstMoment,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> AcceptedFrameFirstMomentState:
        pending = expanded.pending_state
        if not isinstance(pending, _AcceptedFramePendingState):
            raise TypeError("expanded first moment belongs to another component")
        frame = context.geometry.frame(
            context.current_point,
            accepted_displacement,
        )
        numerator = expanded.raw.at(frame.displacement)
        alpha = context.geometry.compress(
            frame=frame,
            pullback_numerator=numerator,
            damping=self.damping,
            rtol=self.rtol,
        )
        return AcceptedFrameFirstMomentState(
            alpha=alpha.detach().clone(),
            frame=frame,
            beta_power=pending.beta_power,
        )
