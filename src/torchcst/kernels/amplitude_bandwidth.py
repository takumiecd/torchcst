"""Amplitude-dependent shared Gaussian bandwidth operator kernels."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel
from .gaussian import Gaussian


class AmplitudeBandwidthSeparable(Kernel):
    r"""A signed rank-one atom whose shared width narrows with ``|w|``.

    One even amplitude gate interpolates the input and output precision between
    the same finite ``sigma_max`` and ``sigma_min`` bounds. The atom row is
    ``(w, input_center, output_center)``.
    """

    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        tau: float = 5e-3,
        temperature: float = 0.25,
        gate_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.profile = Gaussian(sigma_min)
        maximum = self._positive_scalar(sigma_max, name="sigma_max")
        if maximum < self.profile.sigma:
            raise ValueError("sigma_max must not be narrower than sigma_min")
        self.register_buffer("sigma_max", maximum)
        self.register_buffer("tau", self._positive_scalar(tau, name="tau"))
        self.register_buffer(
            "temperature",
            self._positive_scalar(temperature, name="temperature"),
        )
        self.register_buffer(
            "gate_eps",
            self._positive_scalar(gate_eps, name="gate_eps"),
        )

    @property
    def sigma_min(self) -> Tensor:
        return self.profile.sigma

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            1
            + self.profile.parameter_dim(input_chart)
            + self.profile.parameter_dim(output_chart)
        )

    def initialize(
        self,
        input_chart: Chart,
        output_chart: Chart,
        atoms: int,
        *,
        mode: AtomInit,
    ) -> Tensor:
        if mode not in ("balanced", "uniform"):
            raise ValueError("mode must be 'balanced' or 'uniform'")
        input_p = self.profile.initialize(input_chart, atoms, mode="uniform")
        output_p = self.profile.initialize(output_chart, atoms, mode=mode)
        amplitude = input_p.new_empty(atoms, 1)
        amplitude.normal_(mean=0.0, std=0.1 / math.sqrt(atoms))
        return torch.cat((amplitude, input_p, output_p), dim=-1)

    def materialize_atoms(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        phi_input, phi_output = self.factors(input_chart, output_chart, p)
        return torch.einsum("oa,ia->aoi", phi_output, phi_input)

    @property
    def supports_factorization(self) -> bool:
        return True

    def factors(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor]:
        amplitude, input_p, output_p = self._split(input_chart, output_chart, p)
        precision = self.bandwidth_precision(input_chart, output_chart, p)
        phi_input = self.profile.evaluate_with_precision(
            input_chart,
            input_p,
            precision,
        )
        phi_output = self.profile.evaluate_with_precision(
            output_chart,
            output_p,
            precision,
        )
        phi_output = phi_output * amplitude.T
        return phi_input, phi_output

    def amplitude_gate(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the smooth commitment gate for diagnostic use."""

        amplitude, _, _ = self._split(input_chart, output_chart, p)
        magnitude_square = amplitude[:, 0].square() + self.gate_eps.square()
        logit = (magnitude_square.log() - 2.0 * self.tau.log()) / self.temperature
        return torch.sigmoid(logit)

    def bandwidth_precision(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""

        gate = self.amplitude_gate(input_chart, output_chart, p)
        narrow = self.sigma_min.reciprocal().square()
        broad = self.sigma_max.reciprocal().square()
        return broad + (narrow - broad) * gate

    def bandwidth_sigma(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the shared effective input/output sigma for diagnostics."""

        precision = self.bandwidth_precision(input_chart, output_chart, p)
        return precision.clamp_min(torch.finfo(precision.dtype).tiny).rsqrt()

    def _split(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected_dim = self.parameter_dim(input_chart, output_chart)
        if p.ndim != 2 or p.shape[1] != expected_dim:
            raise ValueError(f"p must have shape [atoms, {expected_dim}]")
        input_dim = self.profile.parameter_dim(input_chart)
        input_end = 1 + input_dim
        return p[:, :1], p[:, 1:input_end], p[:, input_end:]

    @staticmethod
    def _positive_scalar(value: float, *, name: str) -> Tensor:
        result = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if result.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        result = result.detach().clone().reshape(())
        if not torch.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def tangent_backend(self, input_chart: Chart, output_chart: Chart):
        from ._tangent import amplitude_bandwidth

        return lambda p: amplitude_bandwidth(self, input_chart, output_chart, p)

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau.item():g}, temperature={self.temperature.item():g}, "
            f"sigma_min={self.sigma_min.item():g}, "
            f"sigma_max={self.sigma_max.item():g}, "
            f"supports_factorization={self.supports_factorization}"
        )
