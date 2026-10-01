"""PyTorch execution for declared amplitude_bandwidth contracts."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState


def amplitude_gate(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> Tensor:
    """Return the interpolating commitment gate for diagnostic use."""

    amplitude, _, _ = _split(state, input_chart, output_chart, p)
    return _interpolating_gate(state, amplitude)


def bandwidth_precision(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> Tensor:
    """Return the shared input/output precision for each atom."""

    amplitude, _, _ = _split(state, input_chart, output_chart, p)
    precision, _ = _precision_and_jacobian(state, amplitude)
    return precision


def bandwidth_sigma(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> Tensor:
    """Return the shared effective input/output sigma for diagnostics."""

    precision = bandwidth_precision(state, input_chart, output_chart, p)
    return precision.clamp_min(torch.finfo(precision.dtype).tiny).rsqrt()


def _split(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    expected_dim = _kernel.parameter_dim(state, input_chart, output_chart)
    if p.ndim != 2 or p.shape[1] != expected_dim:
        raise ValueError(f"p must have shape [atoms, {expected_dim}]")
    input_dim = _profile.parameter_dim(state.profiles[0], input_chart)
    input_end = 1 + input_dim
    return p[:, :1], p[:, 1:input_end], p[:, input_end:]


def _precision_and_jacobian(state, amplitude: Tensor) -> tuple[Tensor, Tensor]:
    if state.setting("law") == "inverse":
        precision, dprecision = _inverse_precision_and_jacobian(state, amplitude)
    else:
        precision, dprecision = _interpolating_precision_and_jacobian(state, amplitude)
    if state.setting("couple_bandwidth"):
        return precision, dprecision
    return precision.detach(), torch.zeros_like(dprecision)


def _magnitude_square(state, amplitude: Tensor) -> Tensor:
    return amplitude[:, 0].square() + state.scalar("gate_eps").square()


def _interpolating_gate(state, amplitude: Tensor) -> Tensor:
    logit = (
        _magnitude_square(state, amplitude).log() - 2.0 * state.scalar("tau").log()
    ) / state.scalar("temperature")
    return torch.sigmoid(logit)


def _interpolating_precision_and_jacobian(
    state,
    amplitude: Tensor,
) -> tuple[Tensor, Tensor]:
    gate = _interpolating_gate(state, amplitude)
    narrow = state.scalar("sigma_min").reciprocal().square()
    broad = state.scalar("sigma_max").reciprocal().square()
    precision = broad + (narrow - broad) * gate
    dprecision = (
        (narrow - broad)
        * gate
        * (1 - gate)
        * (
            2
            * amplitude[:, 0]
            / (state.scalar("temperature") * _magnitude_square(state, amplitude))
        )
    )
    return precision, dprecision


def _inverse_precision_and_jacobian(
    state,
    amplitude: Tensor,
) -> tuple[Tensor, Tensor]:
    log_raw = (
        _magnitude_square(state, amplitude).log() - 2.0 * state.scalar("tau").log()
    )
    log_lo = -2.0 * state.scalar("sigma_max").log()
    log_hi = -2.0 * state.scalar("sigma_min").log()
    inv_temperature = state.scalar("temperature").reciprocal()
    z_lo = (log_raw - log_lo) * inv_temperature
    z_hi = (log_raw - log_hi) * inv_temperature
    log_precision = log_lo + state.scalar("temperature") * (
        F.softplus(z_lo) - F.softplus(z_hi)
    )
    precision = log_precision.exp()
    dprecision = (
        precision
        * (torch.sigmoid(z_lo) - torch.sigmoid(z_hi))
        * (2 * amplitude[:, 0] / _magnitude_square(state, amplitude))
    )
    return precision, dprecision
