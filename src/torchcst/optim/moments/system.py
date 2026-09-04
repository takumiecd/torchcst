"""Composition and atomic state transition for moment components."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from ..atom_grad import AtomGradientObservation, AtomGradRequest
from .base import (
    ExpandedFirstMoment,
    ExpandedSecondMoment,
    FirstMomentComponent,
    MomentContext,
    SecondMomentComponent,
)


@dataclass(frozen=True)
class MomentSystemState:
    """Persistent state owned by both moment components."""

    step: int
    first: object
    second: object


@dataclass(frozen=True)
class ExpandedMoments:
    """Step-local solver views and pending component states."""

    previous_step: int
    next_step: int
    first: ExpandedFirstMoment
    second: ExpandedSecondMoment
    owner_token: object = field(repr=False, compare=False)


class MomentSystem:
    """Combine replaceable first and second moments as one state transition."""

    def __init__(
        self,
        *,
        first: FirstMomentComponent,
        second: SecondMomentComponent,
    ) -> None:
        if not isinstance(first, FirstMomentComponent):
            raise TypeError("first must be a FirstMomentComponent")
        if not isinstance(second, SecondMomentComponent):
            raise TypeError("second must be a SecondMomentComponent")
        self.first = first
        self.second = second
        self._owner_token = object()

    @property
    def observation_request(self) -> AtomGradRequest:
        """The union of observations needed by all installed components."""

        return self.first.observation_request | self.second.observation_request

    def initialize(self, context: MomentContext) -> MomentSystemState:
        return MomentSystemState(
            step=0,
            first=self.first.initialize(context),
            second=self.second.initialize(context),
        )

    def expand(
        self,
        state: MomentSystemState,
        observation: AtomGradientObservation,
        context: MomentContext,
    ) -> ExpandedMoments:
        if not isinstance(state, MomentSystemState):
            raise TypeError("state must be a MomentSystemState")
        if state.step < 0:
            raise ValueError("state step must be nonnegative")
        observation.require(self.observation_request)
        next_step = state.step + 1
        first = self.first.expand(
            state.first,
            observation,
            context,
            next_step=next_step,
        )
        second = self.second.expand(
            state.second,
            observation,
            context,
            next_step=next_step,
        )
        return ExpandedMoments(
            previous_step=state.step,
            next_step=next_step,
            first=first,
            second=second,
            owner_token=self._owner_token,
        )

    def compress(
        self,
        expanded: ExpandedMoments,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> MomentSystemState:
        if expanded.owner_token is not self._owner_token:
            raise ValueError("expanded moments belong to another MomentSystem")
        if expanded.next_step != expanded.previous_step + 1:
            raise ValueError("expanded moment step is inconsistent")
        first = self.first.compress(
            expanded.first,
            accepted_displacement,
            context,
        )
        second = self.second.compress(
            expanded.second,
            accepted_displacement,
            context,
        )
        return MomentSystemState(
            step=expanded.next_step,
            first=first,
            second=second,
        )
