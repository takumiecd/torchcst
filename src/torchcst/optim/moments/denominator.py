"""Reusable quadratic denominator moments for implicit CST optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..atom_grad import AtomGradientObservation, AtomGradRequest


@dataclass(frozen=True)
class DenominatorMomentState:
    """Raw EMA coefficients for a component-wise quadratic denominator."""

    x: Tensor
    y: Tensor
    Z: Tensor
    beta_power: float

    def __post_init__(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [K, P]")
        parameters = self.x.shape[-1]
        if self.y.shape != (*self.x.shape, parameters):
            raise ValueError("y must have shape [K, P, P]")
        if self.Z.shape != (*self.x.shape, parameters, parameters):
            raise ValueError("Z must have shape [K, P, P, P]")
        tensors = (self.x, self.y, self.Z)
        if any(t.device != self.x.device for t in tensors):
            raise ValueError("x, y, and Z must share one device")
        if any(t.dtype != self.x.dtype for t in tensors):
            raise ValueError("x, y, and Z must share one dtype")
        if not math.isfinite(self.beta_power) or not 0.0 <= self.beta_power <= 1.0:
            raise ValueError("beta_power must be finite and lie in [0, 1]")


@dataclass(frozen=True)
class DenominatorPolynomial:
    """A component-wise quadratic polynomial in the local displacement."""

    x: Tensor
    y: Tensor
    Z: Tensor

    def __post_init__(self) -> None:
        DenominatorMomentState(self.x, self.y, self.Z, beta_power=0.0)

    def quadratic_at(self, displacement: Tensor) -> Tensor:
        """Evaluate ``x + 2 y[d] + Z[d,d]`` with shape ``[K, P]``."""

        if displacement.shape != self.x.shape:
            raise ValueError(
                f"displacement must have shape {list(self.x.shape)}"
            )
        if (
            displacement.device != self.x.device
            or displacement.dtype != self.x.dtype
        ):
            raise ValueError("displacement must match denominator device and dtype")
        linear = torch.einsum("kip,kp->ki", self.y, displacement)
        quadratic = torch.einsum(
            "kipq,kp,kq->ki", self.Z, displacement, displacement
        )
        return self.x + 2.0 * linear + quadratic


@dataclass(frozen=True)
class ExpandedDenominatorMoment:
    """Current raw/corrected quadratic denominators and pending state."""

    raw: DenominatorPolynomial
    corrected: DenominatorPolynomial
    pending_state: DenominatorMomentState
    eps: float

    def quadratic_at(self, displacement: Tensor, *, corrected: bool = True) -> Tensor:
        """Evaluate the squared denominator polynomial."""

        return (self.corrected if corrected else self.raw).quadratic_at(displacement)

    def at(self, displacement: Tensor, *, corrected: bool = True) -> Tensor:
        """Evaluate ``sqrt(q(d)) + eps`` component-wise."""

        value = self.quadratic_at(displacement, corrected=corrected)
        return value.clamp_min(0).sqrt() + self.eps


class DenominatorMoment:
    """EMA of ``g²``, ``gH``, and ``HH`` for a component-wise denominator.

    For ``g:[K,P]`` and ``H:[K,P,P]``, the expanded denominator is

    ``D(d) = sqrt(x + 2 y[d] + Z[d,d]) + eps``.

    The output coordinate remains explicit, so ``D(d)`` has shape ``[K, P]``.
    ``Z`` is intentionally materialized here as the complete reference
    representation; lower-memory representations can be added later without
    changing the numerator/denominator interface.
    """

    def __init__(self, beta: float, *, eps: float) -> None:
        if isinstance(beta, bool) or not isinstance(beta, (float, int)):
            raise TypeError("beta must be a real number")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if isinstance(eps, bool) or not isinstance(eps, (float, int)):
            raise TypeError("eps must be a real number")
        if not math.isfinite(float(eps)) or float(eps) < 0.0:
            raise ValueError("eps must be finite and nonnegative")
        self.beta = float(beta)
        self.eps = float(eps)

    @property
    def observation_request(self) -> AtomGradRequest:
        """Observations required to construct the quadratic denominator."""

        return AtomGradRequest(jg=True, gh=True)

    def initialize(self, point: Tensor) -> DenominatorMomentState:
        """Create an all-zero state matching an atom point tensor."""

        if not isinstance(point, Tensor):
            raise TypeError("point must be a torch.Tensor")
        if point.ndim != 2:
            raise ValueError("point must have shape [K, P]")
        if not point.is_floating_point():
            raise ValueError("point must have a floating-point dtype")
        atoms, parameters = point.shape
        return DenominatorMomentState(
            x=point.new_zeros(atoms, parameters),
            y=point.new_zeros(atoms, parameters, parameters),
            Z=point.new_zeros(atoms, parameters, parameters, parameters),
            beta_power=1.0,
        )

    def expand(
        self,
        state: DenominatorMomentState,
        observation: AtomGradientObservation,
        *,
        next_step: int,
    ) -> ExpandedDenominatorMoment:
        """Apply one EMA update and return its quadratic evaluation."""

        if not isinstance(state, DenominatorMomentState):
            raise TypeError("state must be a DenominatorMomentState")
        if isinstance(next_step, bool) or not isinstance(next_step, int):
            raise TypeError("next_step must be an integer")
        if next_step < 1:
            raise ValueError("next_step must be positive")

        expected_beta_power = self.beta ** (next_step - 1)
        if not math.isclose(state.beta_power, expected_beta_power):
            raise ValueError("denominator moment state does not match step or beta")

        if not isinstance(observation, AtomGradientObservation):
            raise TypeError("observation must be an AtomGradientObservation")
        observation.require(self.observation_request)
        assert observation.jg is not None
        assert observation.gh is not None
        self._validate_observation(state, observation.jg, observation.gh)

        g = observation.jg
        H = observation.gh
        current_x = g.square()
        current_y = torch.einsum("ki,kip->kip", g, H)
        current_Z = torch.einsum("kip,kiq->kipq", H, H)

        raw_x = self.beta * state.x + (1.0 - self.beta) * current_x
        raw_y = self.beta * state.y + (1.0 - self.beta) * current_y
        raw_Z = self.beta * state.Z + (1.0 - self.beta) * current_Z
        raw = DenominatorPolynomial(raw_x, raw_y, raw_Z)

        beta_power = state.beta_power * self.beta
        correction = 1.0 / (1.0 - beta_power)
        corrected = DenominatorPolynomial(
            raw_x * correction,
            raw_y * correction,
            raw_Z * correction,
        )
        pending = DenominatorMomentState(
            x=raw_x.detach().clone(),
            y=raw_y.detach().clone(),
            Z=raw_Z.detach().clone(),
            beta_power=beta_power,
        )
        return ExpandedDenominatorMoment(
            raw=raw,
            corrected=corrected,
            pending_state=pending,
            eps=self.eps,
        )

    def compress(
        self,
        expanded: ExpandedDenominatorMoment,
        accepted_displacement: Tensor,
    ) -> DenominatorMomentState:
        """Return the state prepared by ``expand`` without transforming it."""

        if not isinstance(expanded, ExpandedDenominatorMoment):
            raise TypeError("expanded must be an ExpandedDenominatorMoment")
        del accepted_displacement
        return expanded.pending_state

    @staticmethod
    def _validate_observation(
        state: DenominatorMomentState,
        g: Tensor,
        H: Tensor,
    ) -> None:
        if g.shape != state.x.shape:
            raise ValueError("jg must match the denominator moment shape")
        if H.shape != state.y.shape:
            raise ValueError("gh must match the denominator moment shape")
        tensors = (g, H)
        if any(t.device != state.x.device for t in tensors):
            raise ValueError("observations must match the denominator moment device")
        if any(t.dtype != state.x.dtype for t in tensors):
            raise ValueError("observations must match the denominator moment dtype")
