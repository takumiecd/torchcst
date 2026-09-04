"""Common contracts for replaceable CST moment components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from torchcst._derivatives import AffinePullback, FrameGeometry

from ..atom_grad import AtomGradientObservation, AtomGradRequest


class VisibleMetric(Protocol):
    """A positive diagonal action on represented operator values."""

    @property
    def visible_shape(self) -> tuple[int, ...]: ...

    def apply(self, value: Tensor) -> Tensor: ...

    def inner(self, left: Tensor, right: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class MomentContext:
    """Current site geometry and detached atom point."""

    geometry: FrameGeometry
    current_point: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FrameGeometry):
            raise TypeError("geometry must be a FrameGeometry")
        validated = self.geometry.frame(self.current_point).point
        object.__setattr__(self, "current_point", validated)


@dataclass(frozen=True)
class ExpandedFirstMoment:
    """Raw and bias-corrected affine pullbacks for the local objective."""

    raw: AffinePullback
    corrected: AffinePullback
    pending_state: object


@dataclass(frozen=True)
class ExpandedSecondMoment:
    """Bias-corrected metric plus uncommitted raw second-moment state."""

    metric: VisibleMetric
    pending_state: object


class FirstMomentComponent(ABC):
    """Replaceable first-moment expansion and accepted-frame compression."""

    @property
    @abstractmethod
    def observation_request(self) -> AtomGradRequest: ...

    @abstractmethod
    def initialize(self, context: MomentContext) -> object: ...

    @abstractmethod
    def expand(
        self,
        state: object,
        observation: AtomGradientObservation,
        context: MomentContext,
        *,
        next_step: int,
    ) -> ExpandedFirstMoment: ...

    @abstractmethod
    def compress(
        self,
        expanded: ExpandedFirstMoment,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> object: ...


class SecondMomentComponent(ABC):
    """Replaceable second-moment expansion and accepted-state compression."""

    @property
    @abstractmethod
    def observation_request(self) -> AtomGradRequest: ...

    @abstractmethod
    def initialize(self, context: MomentContext) -> object: ...

    @abstractmethod
    def expand(
        self,
        state: object,
        observation: AtomGradientObservation,
        context: MomentContext,
        *,
        next_step: int,
    ) -> ExpandedSecondMoment: ...

    @abstractmethod
    def compress(
        self,
        expanded: ExpandedSecondMoment,
        accepted_displacement: Tensor,
        context: MomentContext,
    ) -> object: ...
