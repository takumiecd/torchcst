"""Reusable affine numerator moments for implicit CST optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._derivatives import AffinePullback

from ..atom_grad import AtomGradientObservation, AtomGradRequest


@dataclass(frozen=True)
class NumeratorMomentState:
    """Raw EMA coefficients for an affine numerator.

    ``m`` is the constant coefficient and ``C`` is the coefficient of the
    local displacement.  Both tensors use the atom-table convention:
    ``m:[K,P]`` and ``C:[K,P,P]``.
    """

    m: Tensor
    C: Tensor
    beta_power: float

    def __post_init__(self) -> None:
        if self.m.ndim != 2:
            raise ValueError("m must have shape [K, P]")
        if self.C.shape != (*self.m.shape, self.m.shape[-1]):
            raise ValueError("C must have shape [K, P, P]")
        if self.m.device != self.C.device or self.m.dtype != self.C.dtype:
            raise ValueError("m and C must share one device and dtype")
        if not math.isfinite(self.beta_power) or not 0.0 <= self.beta_power <= 1.0:
            raise ValueError("beta_power must be finite and lie in [0, 1]")


@dataclass(frozen=True)
class ExpandedNumeratorMoment:
    """The current raw/corrected affine numerator and pending step state."""

    raw: AffinePullback
    corrected: AffinePullback
    pending_state: NumeratorMomentState

    @property
    def point_shape(self) -> tuple[int, int]:
        return tuple(self.corrected.constant.shape)

    @property
    def device(self) -> torch.device:
        return self.corrected.constant.device

    @property
    def dtype(self) -> torch.dtype:
        return self.corrected.constant.dtype

    def at(self, displacement: Tensor, *, corrected: bool = True) -> Tensor:
        """Evaluate the numerator at one local displacement."""

        return (self.corrected if corrected else self.raw).at(displacement)

    def at_zero(self, *, corrected: bool = True) -> Tensor:
        """Evaluate the numerator at zero without contracting its linear term."""

        return (self.corrected if corrected else self.raw).constant


class NumeratorMoment:
    """EMA of ``g`` and optionally ``H`` for a reusable affine numerator.

    Given the current atom observations ``g:[K,P]`` and ``H:[K,P,P]``, the
    expanded numerator is

    ``N(d) = m + C d``.

    ``expand`` computes the next ``m`` and ``C`` values.  The state is kept in
    the fixed atom coordinates, so the accepted displacement does not
    recenter it.  ``compress`` is only the compatibility commit boundary and
    returns the state prepared by ``expand`` unchanged.
    """

    def __init__(self, beta: float, *, include_curvature: bool = True) -> None:
        if isinstance(beta, bool) or not isinstance(beta, (float, int)):
            raise TypeError("beta must be a real number")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if not isinstance(include_curvature, bool):
            raise TypeError("include_curvature must be a bool")
        self.beta = float(beta)
        self.include_curvature = include_curvature

    @property
    def observation_request(self) -> AtomGradRequest:
        """Observations required to construct the affine numerator."""

        return AtomGradRequest(jg=True, gh=self.include_curvature)

    def initialize(self, point: Tensor) -> NumeratorMomentState:
        """Create an all-zero state matching an atom point tensor."""

        if not isinstance(point, Tensor):
            raise TypeError("point must be a torch.Tensor")
        if point.ndim != 2:
            raise ValueError("point must have shape [K, P]")
        if not point.is_floating_point():
            raise ValueError("point must have a floating-point dtype")
        atoms, parameters = point.shape
        return NumeratorMomentState(
            m=point.new_zeros(atoms, parameters),
            C=point.new_zeros(atoms, parameters, parameters),
            beta_power=1.0,
        )

    def expand(
        self,
        state: NumeratorMomentState,
        observation: AtomGradientObservation,
        *,
        next_step: int,
    ) -> ExpandedNumeratorMoment:
        """Apply one EMA update and return its affine evaluation."""

        if not isinstance(state, NumeratorMomentState):
            raise TypeError("state must be a NumeratorMomentState")
        if isinstance(next_step, bool) or not isinstance(next_step, int):
            raise TypeError("next_step must be an integer")
        if next_step < 1:
            raise ValueError("next_step must be positive")

        expected_beta_power = self.beta ** (next_step - 1)
        if not math.isclose(state.beta_power, expected_beta_power):
            raise ValueError("numerator moment state does not match step or beta")

        if not isinstance(observation, AtomGradientObservation):
            raise TypeError("observation must be an AtomGradientObservation")
        observation.require(self.observation_request)
        assert observation.jg is not None
        if self.include_curvature:
            assert observation.gh is not None
            self._validate_observation(state, observation.jg, observation.gh)
            current = AffinePullback(observation.jg, observation.gh)
            previous = AffinePullback(state.m, state.C)
            raw = previous.scaled(self.beta) + current.scaled(1.0 - self.beta)
        else:
            self._validate_gradient(state, observation.jg)
            raw = AffinePullback(
                self.beta * state.m + (1.0 - self.beta) * observation.jg,
                torch.zeros_like(state.C),
            )

        beta_power = state.beta_power * self.beta
        corrected = raw.scaled(1.0 / (1.0 - beta_power))
        pending = NumeratorMomentState(
            m=raw.constant.detach().clone(),
            C=raw.linear.detach().clone(),
            beta_power=beta_power,
        )
        return ExpandedNumeratorMoment(
            raw=raw,
            corrected=corrected,
            pending_state=pending,
        )

    def compress(
        self,
        expanded: ExpandedNumeratorMoment,
        accepted_displacement: Tensor,
    ) -> NumeratorMomentState:
        """Return the state prepared by ``expand`` without transforming it."""

        if not isinstance(expanded, ExpandedNumeratorMoment):
            raise TypeError("expanded must be an ExpandedNumeratorMoment")
        del accepted_displacement
        return expanded.pending_state

    @staticmethod
    def _validate_observation(state: NumeratorMomentState, g: Tensor, H: Tensor) -> None:
        if g.shape != state.m.shape:
            raise ValueError("jg must match the numerator moment shape")
        if H.shape != state.C.shape:
            raise ValueError("gh must match the numerator moment shape")
        if g.device != state.m.device or H.device != state.m.device:
            raise ValueError("observations must match the numerator moment device")
        if g.dtype != state.m.dtype or H.dtype != state.m.dtype:
            raise ValueError("observations must match the numerator moment dtype")

    @staticmethod
    def _validate_gradient(state: NumeratorMomentState, g: Tensor) -> None:
        if g.shape != state.m.shape:
            raise ValueError("jg must match the numerator moment shape")
        if g.device != state.m.device:
            raise ValueError("observation must match the numerator moment device")
        if g.dtype != state.m.dtype:
            raise ValueError("observation must match the numerator moment dtype")
