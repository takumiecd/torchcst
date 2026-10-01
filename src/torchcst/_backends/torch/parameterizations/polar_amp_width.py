"""PyTorch execution for declared polar_amplitude_bandwidth contracts."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry import Chart


def amplitude(state, input_chart: Chart, output_chart: Chart, p: Tensor) -> Tensor:
    """Return the bounded signed amplitude represented by each atom."""

    polar, _, _ = _split(state, input_chart, output_chart, p)
    amplitude, _ = _amplitude_and_alpha(state, polar)
    return amplitude


def bandwidth_alpha(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    """Return the radial interpolation coordinate in ``[0, 1]``."""

    polar, _, _ = _split(state, input_chart, output_chart, p)
    _, alpha = _amplitude_and_alpha(state, polar)
    return alpha


def bandwidth_bounds(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(L(w), U(w))`` for diagnostic use."""

    (input_bounds, output_bounds) = bandwidth_bounds_by_side(
        state, input_chart, output_chart, p
    )
    _require_shared_bandwidths(state)
    del output_bounds
    return input_bounds


def bandwidth_bounds_by_side(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Return ``((L_in, U_in), (L_out, U_out))``."""

    polar, _, _ = _split(state, input_chart, output_chart, p)
    amplitude, _ = _amplitude_and_alpha(state, polar)
    _, lower_input, upper_input = _sigma_bounds(state, amplitude, side="input")
    _, lower_output, upper_output = _sigma_bounds(state, amplitude, side="output")
    return (lower_input, upper_input), (lower_output, upper_output)


def bandwidth_sigma(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    """Return the shared effective input/output sigma for each atom."""

    sigma_input, _ = bandwidth_sigmas(state, input_chart, output_chart, p)
    _require_shared_bandwidths(state)
    return sigma_input


def bandwidth_sigmas(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return effective ``(sigma_input, sigma_output)`` for each atom."""

    polar, _, _ = _split(state, input_chart, output_chart, p)
    amplitude, alpha = _amplitude_and_alpha(state, polar)
    return _bandwidth_sigmas(state, amplitude, alpha)


def bandwidth_precision(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> Tensor:
    """Return the shared input/output precision for each atom."""

    sigma = bandwidth_sigma(state, input_chart, output_chart, p)
    return sigma.reciprocal().square()


def bandwidth_precisions(
    state, input_chart: Chart, output_chart: Chart, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return input and output precisions for split bandwidths."""

    sigma_input, sigma_output = bandwidth_sigmas(state, input_chart, output_chart, p)
    return sigma_input.reciprocal().square(), sigma_output.reciprocal().square()


def _split(
    state,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    expected_dim = _kernel.parameter_dim(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = _profile.parameter_dim(state.profiles[0], input_chart)
    input_end = 2 + input_dim
    return p[:, :2], p[:, 2:input_end], p[:, input_end:]


def _amplitude_and_alpha(state, polar: Tensor) -> tuple[Tensor, Tensor]:
    radius_square = polar.square().sum(dim=-1)
    # The exact polar map is used everywhere except the singular origin.
    safe_square = radius_square.clamp_min(torch.finfo(polar.dtype).tiny)
    amplitude = (
        state.scalar("amplitude_max").to(polar) * polar[:, 0] / safe_square.sqrt()
    )
    alpha = ((radius_square - 1.0) / 3.0).clamp(0.0, 1.0)
    return amplitude, alpha


def _bandwidth_sigmas(state, amplitude: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
    sigma_input, _, _ = _sigma_bounds(state, amplitude, alpha, side="input")
    sigma_output, _, _ = _sigma_bounds(state, amplitude, alpha, side="output")
    return sigma_input, sigma_output


def _sigma_bounds(
    state,
    amplitude: Tensor,
    alpha: Tensor | None = None,
    *,
    side: Literal["input", "output"] = "input",
) -> tuple[Tensor, Tensor, Tensor]:
    if side == "input":
        minimum = state.scalar("sigma_min_input")
        birth = state.scalar("sigma_birth_input")
        maximum = state.scalar("sigma_max_input")
        floor = state.scalar("upper_floor_input")
    elif side == "output":
        minimum = state.scalar("sigma_min_output")
        birth = state.scalar("sigma_birth_output")
        maximum = state.scalar("sigma_max_output")
        floor = state.scalar("upper_floor_output")
    else:
        raise ValueError("side must be 'input' or 'output'")
    x = (amplitude / state.scalar("w_c").to(amplitude)).square()
    lower_delta = birth.to(amplitude) - minimum.to(amplitude)
    upper_delta = maximum.to(amplitude) - minimum.to(amplitude)
    kappa = state.scalar("kappa").to(amplitude)
    upper_x = x.pow(state.scalar("upper_decay_power").to(amplitude))
    upper = minimum.to(amplitude) + upper_delta * kappa / (kappa + upper_x)
    # Keep radial activity meaningful for high-amplitude atoms.  Without
    # this floor U(w) converges to sigma_min, so alpha loses all authority
    # precisely when a strong atom becomes trapped on a single site.
    upper = torch.maximum(upper, floor.to(amplitude))
    lower = minimum.to(amplitude) + lower_delta / (
        1.0 + state.scalar("lower_kappa").to(amplitude) * x
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


def _require_shared_bandwidths(state) -> None:
    pairs = (
        (state.scalar("sigma_min_input"), state.scalar("sigma_min_output")),
        (state.scalar("sigma_birth_input"), state.scalar("sigma_birth_output")),
        (state.scalar("sigma_max_input"), state.scalar("sigma_max_output")),
        (state.scalar("upper_floor_input"), state.scalar("upper_floor_output")),
    )
    if not all(bool(torch.equal(left, right)) for left, right in pairs):
        raise ValueError("input and output bandwidths differ; use the by-side API")
