"""Current-tangent first moment, independent of accepted Taylor frames."""

import math
from dataclasses import dataclass

import torch

from torchcst._derivatives import AffinePullback, RepresentationFrame

from ..atom_grad import AtomGradRequest
from .base import ExpandedFirstMoment, FirstMomentComponent


@dataclass(frozen=True)
class TangentFirstMomentState:
    alpha: torch.Tensor
    frame: RepresentationFrame
    beta_power: float


class TangentFirstMoment(FirstMomentComponent):
    def __init__(self, beta, *, damping=0.0):
        self.beta = beta
        self.damping = damping

    @property
    def observation_request(self):
        return AtomGradRequest(jg=True)

    def initialize(self, context):
        return TangentFirstMomentState(
            torch.zeros_like(context.current_point),
            context.geometry.frame(context.current_point),
            1.0,
        )

    def expand(self, state, observation, context, *, next_step):
        if not isinstance(state, TangentFirstMomentState):
            raise TypeError("expected a tangent first-moment state")
        if not math.isclose(state.beta_power, self.beta ** (next_step - 1)):
            raise ValueError("first-moment step or beta mismatch")
        observation.require(self.observation_request)
        point = context.current_point
        old = (
            torch.zeros_like(point)
            if next_step == 1
            else context.geometry.pullback_from_frame(
                current_point=point,
                source_frame=state.frame,
                source_coefficients=state.alpha,
            ).constant
        )
        b = self.beta * old + (1 - self.beta) * observation.jg
        raw = AffinePullback(b, point.new_zeros(*point.shape, point.shape[-1]))
        power = self.beta**next_step
        return ExpandedFirstMoment(raw, raw.scaled(1 / (1 - power)), power)

    def compress(self, expanded, accepted_displacement, context):
        frame = context.geometry.frame(context.current_point)
        alpha = context.geometry.compress(
            frame=frame, pullback_numerator=expanded.raw.constant, damping=self.damping
        )
        return TangentFirstMomentState(alpha.detach(), frame, expanded.pending_state)
