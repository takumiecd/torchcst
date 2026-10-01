"""PyTorch execution for declared direct_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst._backends.torch.parameterizations import (
    direct_amp_width as _parameterizations,
)
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState


def apply_parameter_update(
    state,
    input_chart: ChartState,
    output_chart: ChartState | Tensor,
    p: Tensor,
    displacement: Tensor | None = None,
    *,
    step_size: float,
) -> Tensor:
    """Apply a direct-amplitude proposal and advance its activity state."""

    if displacement is None:
        if not isinstance(output_chart, Tensor):
            raise TypeError("single-chart update requires atom parameters")
        return _single_apply_update(state, input_chart, output_chart, p, step_size)
    direct, input_p, output_p = _parameterizations._split(
        state, input_chart, output_chart, p
    )
    if displacement.shape != p.shape:
        raise ValueError("displacement must match the atom parameter shape")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")

    maximum = state.scalar("amplitude_max").to(p)
    old_w = direct[:, 0].clamp(-maximum, maximum)
    old_q = direct[:, 1].clamp(1.0, 4.0)
    accepted_w = (old_w + displacement[:, 0]).clamp(-maximum, maximum)

    accepted_delta = (accepted_w - old_w) / maximum
    delta_square = old_q * accepted_delta.square()
    geometric_q = (old_q + delta_square).clamp(1.0, 4.0)

    # One ordinary gradient step for R(q)=lambda/2*(q-1)^2. The clamp
    # retains the state contract even for an unusually large step size.
    regularized_q = (
        geometric_q
        - state.scalar("radial_regularization").to(p)
        * p.new_tensor(step_size)
        * (geometric_q - 1.0)
    ).clamp(1.0, 4.0)

    _, input_d, output_d = _parameterizations._split(
        state, input_chart, output_chart, displacement
    )
    updated_input = _profile.apply_parameter_update(
        state.profiles[0], input_chart, input_p, input_d
    )
    updated_output = _profile.apply_parameter_update(
        state.profiles[1], output_chart, output_p, output_d
    )
    updated_direct = torch.stack((accepted_w, regularized_q), dim=-1)
    return torch.cat((updated_direct, updated_input, updated_output), dim=-1)


def project_parameter_gradient(
    state,
    input_chart: ChartState,
    output_chart: ChartState | Tensor,
    p: Tensor,
    gradient: Tensor | None = None,
) -> Tensor:
    if gradient is None:
        if not isinstance(output_chart, Tensor):
            raise TypeError("single-chart projection requires atom parameters")
        direct, center = _parameterizations._single_split(
            state, input_chart, output_chart
        )
        direct_g, center_g = _parameterizations._single_split(state, input_chart, p)
        return torch.cat(
            (
                torch.stack((direct_g[:, 0], torch.zeros_like(direct_g[:, 1])), dim=-1),
                _profile.project_gradient(
                    state.profiles[0], input_chart, center, center_g
                ),
            ),
            dim=-1,
        )
    direct, input_p, output_p = _parameterizations._split(
        state, input_chart, output_chart, p
    )
    del direct
    direct_g, input_g, output_g = _parameterizations._split(
        state, input_chart, output_chart, gradient
    )
    projected_direct = torch.stack(
        (direct_g[:, 0], torch.zeros_like(direct_g[:, 1])),
        dim=-1,
    )
    return torch.cat(
        (
            projected_direct,
            _profile.project_gradient(state.profiles[0], input_chart, input_p, input_g),
            _profile.project_gradient(
                state.profiles[1], output_chart, output_p, output_g
            ),
        ),
        dim=-1,
    )


def transport_parameter_state(
    kernel_state,
    input_chart: ChartState,
    output_chart: ChartState | Tensor,
    old: Tensor,
    new: Tensor,
    state: Tensor | None = None,
) -> Tensor:
    if state is None:
        if not isinstance(output_chart, Tensor):
            raise TypeError("single-chart transport requires atom parameters")
        _, old_center = _parameterizations._single_split(
            kernel_state, input_chart, output_chart
        )
        _, new_center = _parameterizations._single_split(kernel_state, input_chart, old)
        direct_state, center_state = _parameterizations._single_split(
            kernel_state, input_chart, new
        )
        return torch.cat(
            (
                direct_state,
                _profile.transport_state(
                    kernel_state.profiles[0],
                    input_chart,
                    old_center,
                    new_center,
                    center_state,
                ),
            ),
            dim=-1,
        )
    _, old_i, old_o = _parameterizations._split(
        kernel_state, input_chart, output_chart, old
    )
    _, new_i, new_o = _parameterizations._split(
        kernel_state, input_chart, output_chart, new
    )
    direct_s, state_i, state_o = _parameterizations._split(
        kernel_state, input_chart, output_chart, state
    )
    return torch.cat(
        (
            direct_s,
            _profile.transport_state(
                kernel_state.profiles[0], input_chart, old_i, new_i, state_i
            ),
            _profile.transport_state(
                kernel_state.profiles[1], output_chart, old_o, new_o, state_o
            ),
        ),
        dim=-1,
    )


def _single_apply_update(
    state, chart: ChartState, p: Tensor, displacement: Tensor, step_size: float
) -> Tensor:
    direct, center = _parameterizations._single_split(state, chart, p)
    if displacement.shape != p.shape:
        raise ValueError("displacement must match the atom parameter shape")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    maximum = state.scalar("amplitude_max").to(p)
    old_w = direct[:, 0].clamp(-maximum, maximum)
    old_q = direct[:, 1].clamp(1.0, 4.0)
    accepted_w = (old_w + displacement[:, 0]).clamp(-maximum, maximum)
    delta_square = old_q * ((accepted_w - old_w) / maximum).square()
    geometric_q = (old_q + delta_square).clamp(1.0, 4.0)
    regularized_q = (
        geometric_q
        - state.scalar("radial_regularization").to(p)
        * p.new_tensor(step_size)
        * (geometric_q - 1.0)
    ).clamp(1.0, 4.0)
    updated_center = _profile.apply_parameter_update(
        state.profiles[0], chart, center, displacement[:, 2:]
    )
    return torch.cat(
        (torch.stack((accepted_w, regularized_q), dim=-1), updated_center),
        dim=-1,
    )
