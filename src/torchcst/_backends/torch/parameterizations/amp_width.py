"""PyTorch execution for legacy amplitude_bandwidth contracts."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst.geometry import Chart


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
        precision, dprecision = self._inverse_precision_and_jacobian(amplitude)
    else:
        precision, dprecision = self._interpolating_precision_and_jacobian(amplitude)
    if self.couple_bandwidth:
        return precision, dprecision
    return precision.detach(), torch.zeros_like(dprecision)


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
    log_precision = log_lo + self.temperature * (F.softplus(z_lo) - F.softplus(z_hi))
    precision = log_precision.exp()
    dprecision = (
        precision
        * (torch.sigmoid(z_lo) - torch.sigmoid(z_hi))
        * (2 * amplitude[:, 0] / self._magnitude_square(amplitude))
    )
    return precision, dprecision
