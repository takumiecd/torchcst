"""Independent numerator/denominator state for normalized CST updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..atom_grad import AtomGradientObservation, AtomGradRequest
from .base import MomentContext
from .denominator import (
    DenominatorMoment,
    DenominatorMomentState,
    ExpandedDenominatorMoment,
)
from .numerator import ExpandedNumeratorMoment, NumeratorMoment, NumeratorMomentState


@dataclass(frozen=True)
class UnitDenominatorState:
    """Shape metadata for the stateless denominator ``D(d)=1``."""

    point_shape: tuple[int, int]


@dataclass(frozen=True)
class ExpandedUnitDenominator:
    """Candidate evaluation for a unit denominator."""

    point_shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype
    pending_state: UnitDenominatorState

    def at(self, displacement: Tensor) -> Tensor:
        if displacement.shape != self.point_shape:
            raise ValueError(f"displacement must have shape {list(self.point_shape)}")
        if displacement.device != self.device or displacement.dtype != self.dtype:
            raise ValueError("displacement must match denominator device and dtype")
        return torch.ones_like(displacement)


class UnitDenominator:
    """Stateless denominator used by SGD and Momentum wrappers."""

    observation_request = AtomGradRequest()

    def initialize(self, point: Tensor) -> UnitDenominatorState:
        if not isinstance(point, Tensor):
            raise TypeError("point must be a torch.Tensor")
        if point.ndim != 2:
            raise ValueError("point must have shape [K, P]")
        return UnitDenominatorState(tuple(point.shape))

    def expand(
        self,
        state: UnitDenominatorState,
        observation: AtomGradientObservation,
        *,
        next_step: int,
    ) -> ExpandedUnitDenominator:
        if not isinstance(state, UnitDenominatorState):
            raise TypeError("state must be a UnitDenominatorState")
        if isinstance(next_step, bool) or not isinstance(next_step, int) or next_step < 1:
            raise ValueError("next_step must be a positive integer")
        observation.require(self.observation_request)
        if observation.jg is not None:
            if tuple(observation.jg.shape) != state.point_shape:
                raise ValueError("observation shape does not match unit denominator")
            return ExpandedUnitDenominator(
                point_shape=state.point_shape,
                device=observation.jg.device,
                dtype=observation.jg.dtype,
                pending_state=state,
            )
        raise ValueError("unit denominator requires an observation with jg")

    def compress(
        self,
        expanded: ExpandedUnitDenominator,
        accepted_displacement: Tensor,
    ) -> UnitDenominatorState:
        if not isinstance(expanded, ExpandedUnitDenominator):
            raise TypeError("expanded must be an ExpandedUnitDenominator")
        del accepted_displacement
        return expanded.pending_state


@dataclass(frozen=True)
class NormalizedMomentSystemState:
    """Persistent step counter plus independent N and D component states."""

    step: int
    numerator: NumeratorMomentState
    denominator: DenominatorMomentState | UnitDenominatorState


@dataclass(frozen=True)
class ExpandedNormalizedMoments:
    """Step-local numerator/denominator evaluations and pending states."""

    previous_step: int
    next_step: int
    numerator: ExpandedNumeratorMoment
    denominator: ExpandedDenominatorMoment | ExpandedUnitDenominator
    owner_token: object


class NormalizedMomentSystem:
    """Compose reusable N and D components without moving either state frame."""

    def __init__(self, *, numerator: NumeratorMoment, denominator) -> None:
        if not isinstance(numerator, NumeratorMoment):
            raise TypeError("numerator must be a NumeratorMoment")
        if not isinstance(denominator, (DenominatorMoment, UnitDenominator)):
            raise TypeError("denominator must be a DenominatorMoment or UnitDenominator")
        self.numerator = numerator
        self.denominator = denominator
        self._owner_token = object()

    @property
    def observation_request(self) -> AtomGradRequest:
        return self.numerator.observation_request | self.denominator.observation_request

    def initialize(self, context: MomentContext) -> NormalizedMomentSystemState:
        point = context.current_point
        return NormalizedMomentSystemState(
            step=0,
            numerator=self.numerator.initialize(point),
            denominator=self.denominator.initialize(point),
        )

    def expand(
        self,
        state: NormalizedMomentSystemState,
        observation: AtomGradientObservation,
        context: MomentContext,
    ) -> ExpandedNormalizedMoments:
        if not isinstance(state, NormalizedMomentSystemState):
            raise TypeError("state must be a NormalizedMomentSystemState")
        if state.step < 0:
            raise ValueError("state step must be nonnegative")
        del context
        observation.require(self.observation_request)
        next_step = state.step + 1
        numerator = self.numerator.expand(
            state.numerator,
            observation,
            next_step=next_step,
        )
        denominator = self.denominator.expand(
            state.denominator,
            observation,
            next_step=next_step,
        )
        return ExpandedNormalizedMoments(
            previous_step=state.step,
            next_step=next_step,
            numerator=numerator,
            denominator=denominator,
            owner_token=self._owner_token,
        )

    def compress(
        self,
        expanded: ExpandedNormalizedMoments,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> NormalizedMomentSystemState:
        del context
        if expanded.owner_token is not self._owner_token:
            raise ValueError("expanded moments belong to another NormalizedMomentSystem")
        if expanded.next_step != expanded.previous_step + 1:
            raise ValueError("expanded moment step is inconsistent")
        numerator = self.numerator.compress(
            expanded.numerator,
            accepted_displacement,
        )
        denominator = self.denominator.compress(
            expanded.denominator,
            accepted_displacement,
        )
        return NormalizedMomentSystemState(
            step=expanded.next_step,
            numerator=numerator,
            denominator=denominator,
        )
