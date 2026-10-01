"""PyTorch execution for legacy polar_amplitude_bandwidth contracts."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchcst.geometry import Chart


def amplitude(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
    """Return the bounded signed amplitude represented by each atom."""

    polar, _, _ = self._split(input_chart, output_chart, p)
    amplitude, _ = self._amplitude_and_alpha(polar)
    return amplitude


def bandwidth_alpha(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
    """Return the radial interpolation coordinate in ``[0, 1]``."""

    polar, _, _ = self._split(input_chart, output_chart, p)
    _, alpha = self._amplitude_and_alpha(polar)
    return alpha


def bandwidth_bounds(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(L(w), U(w))`` for diagnostic use."""

    (input_bounds, output_bounds) = self.bandwidth_bounds_by_side(
        input_chart, output_chart, p
    )
    self._require_shared_bandwidths()
    del output_bounds
    return input_bounds


def bandwidth_bounds_by_side(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Return ``((L_in, U_in), (L_out, U_out))``."""

    polar, _, _ = self._split(input_chart, output_chart, p)
    amplitude, _ = self._amplitude_and_alpha(polar)
    _, lower_input, upper_input = self._sigma_bounds(amplitude, side="input")
    _, lower_output, upper_output = self._sigma_bounds(amplitude, side="output")
    return (lower_input, upper_input), (lower_output, upper_output)


def bandwidth_sigma(self, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
    """Return the shared effective input/output sigma for each atom."""

    sigma_input, _ = self.bandwidth_sigmas(input_chart, output_chart, p)
    self._require_shared_bandwidths()
    return sigma_input


def bandwidth_sigmas(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return effective ``(sigma_input, sigma_output)`` for each atom."""

    polar, _, _ = self._split(input_chart, output_chart, p)
    amplitude, alpha = self._amplitude_and_alpha(polar)
    return self._bandwidth_sigmas(amplitude, alpha)


def bandwidth_precision(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    """Return the shared input/output precision for each atom."""

    sigma = self.bandwidth_sigma(input_chart, output_chart, p)
    return sigma.reciprocal().square()


def bandwidth_precisions(
    self, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return input and output precisions for split bandwidths."""

    sigma_input, sigma_output = self.bandwidth_sigmas(input_chart, output_chart, p)
    return sigma_input.reciprocal().square(), sigma_output.reciprocal().square()


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
    input_end = 2 + input_dim
    return p[:, :2], p[:, 2:input_end], p[:, input_end:]


def _amplitude_and_alpha(self, polar: Tensor) -> tuple[Tensor, Tensor]:
    radius_square = polar.square().sum(dim=-1)
    # The exact polar map is used everywhere except the singular origin.
    safe_square = radius_square.clamp_min(torch.finfo(polar.dtype).tiny)
    amplitude = self.amplitude_max.to(polar) * polar[:, 0] / safe_square.sqrt()
    alpha = ((radius_square - 1.0) / 3.0).clamp(0.0, 1.0)
    return amplitude, alpha


def _bandwidth_sigmas(self, amplitude: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
    sigma_input, _, _ = self._sigma_bounds(amplitude, alpha, side="input")
    sigma_output, _, _ = self._sigma_bounds(amplitude, alpha, side="output")
    return sigma_input, sigma_output


def _sigma_bounds(
    self,
    amplitude: Tensor,
    alpha: Tensor | None = None,
    *,
    side: Literal["input", "output"] = "input",
) -> tuple[Tensor, Tensor, Tensor]:
    if side == "input":
        minimum = self.sigma_min_input
        birth = self.sigma_birth_input
        maximum = self.sigma_max_input
        floor = self.upper_floor_input
    elif side == "output":
        minimum = self.sigma_min_output
        birth = self.sigma_birth_output
        maximum = self.sigma_max_output
        floor = self.upper_floor_output
    else:
        raise ValueError("side must be 'input' or 'output'")
    x = (amplitude / self.w_c.to(amplitude)).square()
    lower_delta = birth.to(amplitude) - minimum.to(amplitude)
    upper_delta = maximum.to(amplitude) - minimum.to(amplitude)
    kappa = self.kappa.to(amplitude)
    upper_x = x.pow(self.upper_decay_power.to(amplitude))
    upper = minimum.to(amplitude) + upper_delta * kappa / (kappa + upper_x)
    # Keep radial activity meaningful for high-amplitude atoms.  Without
    # this floor U(w) converges to sigma_min, so alpha loses all authority
    # precisely when a strong atom becomes trapped on a single site.
    upper = torch.maximum(upper, floor.to(amplitude))
    lower = minimum.to(amplitude) + lower_delta / (
        1.0 + self.lower_kappa.to(amplitude) * x
    )
    # Independent lower decay and upper decay can otherwise cross. The
    # effective upper envelope must never narrow below the lower curve.
    upper = torch.maximum(upper, lower)
    if alpha is None:
        sigma = lower
    else:
        # Width is a scale, so alpha advances a constant fraction of the
        # multiplicative range rather than a constant absolute distance.
        sigma = torch.exp((1.0 - alpha) * lower.log() + alpha * upper.log())
        # exp(log(bound)) can round one ulp outside its closed interval.
        sigma = sigma.clamp(min=lower, max=upper)
    return sigma, lower, upper


def _require_shared_bandwidths(self) -> None:
    pairs = (
        (self.sigma_min_input, self.sigma_min_output),
        (self.sigma_birth_input, self.sigma_birth_output),
        (self.sigma_max_input, self.sigma_max_output),
        (self.upper_floor_input, self.upper_floor_output),
    )
    if not all(bool(torch.equal(left, right)) for left, right in pairs):
        raise ValueError("input and output bandwidths differ; use the by-side API")
