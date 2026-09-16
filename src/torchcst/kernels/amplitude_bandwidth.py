"""Amplitude-dependent shared Gaussian bandwidth operator kernels."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.geometry import Chart

from .base import AtomInit, Kernel, Profile
from .gaussian import Gaussian

_LAWS = ("interpolating", "inverse")


class AmplitudeBandwidthSeparable(Kernel):
    r"""A signed rank-one atom whose shared width narrows with ``|w|``.

    ``law="interpolating"`` uses one even amplitude gate to interpolate the
    shared input and output precision between finite ``sigma_max`` and
    ``sigma_min``. ``law="inverse"`` uses the smooth map ``σ ≈ τ / |w|``,
    softly clamped to the same bounds in log-precision. The atom row is
    ``(w, input_center, output_center)``. The default profile is ``Gaussian``;
    compact profiles such as ``WendlandC2`` and ``Triweight`` use the same
    precision convention, with ``sigma`` as the support radius.
    """

    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        tau: float = 5e-3,
        temperature: float = 0.25,
        gate_eps: float = 1e-12,
        law: str = "interpolating",
        profile: Profile | None = None,
    ) -> None:
        super().__init__()
        if law not in _LAWS:
            raise ValueError("law must be 'interpolating' or 'inverse'")
        self.law = law
        if profile is None:
            profile = Gaussian(sigma_min)
        elif not isinstance(profile, Profile):
            raise TypeError("profile must implement the Profile contract")
        if not hasattr(profile, "evaluate_with_precision") or not hasattr(
            profile, "tangent_with_precision"
        ):
            raise TypeError(
                "bandwidth profile must implement evaluate_with_precision "
                "and tangent_with_precision"
            )
        if not hasattr(profile, "sigma"):
            raise TypeError("bandwidth profile must expose a sigma buffer")
        minimum = self._positive_scalar(sigma_min, name="sigma_min")
        if not torch.allclose(
            profile.sigma.detach().to(dtype=minimum.dtype).reshape(()),
            minimum,
        ):
            raise ValueError("profile.sigma must match sigma_min")
        self.profile = profile
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
        precision, _ = self._precision_and_jacobian(amplitude)
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
        """Return the interpolating commitment gate for diagnostic use."""

        amplitude, _, _ = self._split(input_chart, output_chart, p)
        return self._interpolating_gate(amplitude)

    def bandwidth_precision(
        self,
        input_chart: Chart,
        output_chart: Chart,
        p: Tensor,
    ) -> Tensor:
        """Return the shared input/output precision for each atom."""

        amplitude, _, _ = self._split(input_chart, output_chart, p)
        precision, _ = self._precision_and_jacobian(amplitude)
        return precision

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

    def _precision_and_jacobian(self, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        if self.law == "inverse":
            return self._inverse_precision_and_jacobian(amplitude)
        return self._interpolating_precision_and_jacobian(amplitude)

    def _magnitude_square(self, amplitude: Tensor) -> Tensor:
        return amplitude[:, 0].square() + self.gate_eps.square()

    def _interpolating_gate(self, amplitude: Tensor) -> Tensor:
        logit = (
            self._magnitude_square(amplitude).log() - 2.0 * self.tau.log()
        ) / self.temperature
        return torch.sigmoid(logit)

    def _interpolating_precision_and_jacobian(
        self,
        amplitude: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gate = self._interpolating_gate(amplitude)
        narrow = self.sigma_min.reciprocal().square()
        broad = self.sigma_max.reciprocal().square()
        precision = broad + (narrow - broad) * gate
        dprecision = (
            (narrow - broad)
            * gate
            * (1 - gate)
            * (2 * amplitude[:, 0] / (self.temperature * self._magnitude_square(amplitude)))
        )
        return precision, dprecision

    def _inverse_precision_and_jacobian(
        self,
        amplitude: Tensor,
    ) -> tuple[Tensor, Tensor]:
        log_raw = self._magnitude_square(amplitude).log() - 2.0 * self.tau.log()
        log_lo = -2.0 * self.sigma_max.log()
        log_hi = -2.0 * self.sigma_min.log()
        inv_temperature = self.temperature.reciprocal()
        z_lo = (log_raw - log_lo) * inv_temperature
        z_hi = (log_raw - log_hi) * inv_temperature
        log_precision = log_lo + self.temperature * (
            F.softplus(z_lo) - F.softplus(z_hi)
        )
        precision = log_precision.exp()
        dprecision = (
            precision
            * (torch.sigmoid(z_lo) - torch.sigmoid(z_hi))
            * (2 * amplitude[:, 0] / self._magnitude_square(amplitude))
        )
        return precision, dprecision

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
            f"law={self.law}, "
            f"tau={self.tau.item():g}, temperature={self.temperature.item():g}, "
            f"profile={type(self.profile).__name__}, "
            f"sigma_min={self.sigma_min.item():g}, "
            f"sigma_max={self.sigma_max.item():g}, "
            f"supports_factorization={self.supports_factorization}"
        )
