"""Amplitude-dependent Gaussian bandwidth operator kernels."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian


class AmplitudeBandwidthSeparable(Kernel):
    r"""A signed rank-one atom whose output width narrows with ``|w|``.

    The output precision interpolates smoothly between ``sigma_explore`` and
    ``output_profile.sigma`` using an even amplitude gate. The atom row is
    ``(w, input_profile_coordinates, output_center)``.
    """

    def __init__(
        self,
        *,
        input_profile: Profile,
        output_profile: Gaussian,
        tau: float = 5e-3,
        temperature: float = 0.25,
        sigma_explore: float = math.inf,
        gate_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not isinstance(input_profile, Profile):
            raise TypeError("input_profile must be a Profile")
        if not isinstance(output_profile, Gaussian):
            raise TypeError("output_profile must be a Gaussian")
        self.input_profile = input_profile
        self.output_profile = output_profile
        self.register_buffer("tau", self._positive_scalar(tau, name="tau"))
        self.register_buffer(
            "temperature",
            self._positive_scalar(temperature, name="temperature"),
        )
        self.register_buffer(
            "gate_eps",
            self._positive_scalar(gate_eps, name="gate_eps"),
        )
        explore = torch.as_tensor(sigma_explore, dtype=torch.get_default_dtype())
        if explore.numel() != 1:
            raise ValueError("sigma_explore must be a scalar")
        explore = explore.detach().clone().reshape(())
        if torch.isnan(explore) or explore <= 0:
            raise ValueError("sigma_explore must be positive and not NaN")
        if explore < output_profile.sigma:
            raise ValueError("sigma_explore must not be narrower than output sigma")
        self.register_buffer("sigma_explore", explore)

    def parameter_dim(self, input_chart: Chart, output_chart: Chart) -> int:
        return (
            1
            + self.input_profile.parameter_dim(input_chart)
            + self.output_profile.parameter_dim(output_chart)
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
        input_p = self.input_profile.initialize(input_chart, atoms, mode="uniform")
        output_p = self.output_profile.initialize(output_chart, atoms, mode=mode)
        amplitude = input_p.new_empty(atoms, 1)
        amplitude.normal_(mean=0.0, std=1.0 / math.sqrt(atoms))
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
        phi_input = self.input_profile.evaluate(input_chart, input_p)
        precision = self.output_precision(input_chart, output_chart, p)
        phi_output = self.output_profile.evaluate_with_precision(
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
        logit = (
            magnitude_square.log() - 2.0 * self.tau.log()
        ) / self.temperature
        return torch.sigmoid(logit)

    def output_precision(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return one amplitude-dependent output precision per atom."""

        gate = self.amplitude_gate(input_chart, output_chart, p)
        narrow = self.output_profile.sigma.reciprocal().square()
        explore = torch.where(
            torch.isinf(self.sigma_explore),
            torch.zeros_like(self.sigma_explore),
            self.sigma_explore.reciprocal().square(),
        )
        return explore + (narrow - explore) * gate

    def output_sigma(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the effective output sigma for diagnostic use."""

        precision = self.output_precision(input_chart, output_chart, p)
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
        input_dim = self.input_profile.parameter_dim(input_chart)
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

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau.item():g}, temperature={self.temperature.item():g}, "
            f"sigma_explore={self.sigma_explore.item():g}, "
            f"supports_factorization={self.supports_factorization}"
        )
