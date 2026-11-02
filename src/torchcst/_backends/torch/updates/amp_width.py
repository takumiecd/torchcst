"""PyTorch execution for declared amplitude_bandwidth contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.parameterizations import amp_width as _parameterizations
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import ChartState


def project_parameter_gradient(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    _, output_p, input_p = _parameterizations._split(
        state, output_chart, input_chart, p
    )
    amplitude_g, output_g, input_g = _parameterizations._split(
        state, output_chart, input_chart, gradient
    )
    return torch.cat(
        (
            amplitude_g,
            _profile.project_gradient(
                state.profiles[0], output_chart, output_p, output_g
            ),
            _profile.project_gradient(state.profiles[1], input_chart, input_p, input_g),
        ),
        dim=-1,
    )


def apply_parameter_update(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    del step_size
    amplitude, output_p, input_p = _parameterizations._split(
        state, output_chart, input_chart, p
    )
    amplitude_d, output_d, input_d = _parameterizations._split(
        state, output_chart, input_chart, displacement
    )
    return torch.cat(
        (
            amplitude + amplitude_d,
            _profile.apply_parameter_update(
                state.profiles[0], output_chart, output_p, output_d
            ),
            _profile.apply_parameter_update(
                state.profiles[1], input_chart, input_p, input_d
            ),
        ),
        dim=-1,
    )


def transport_parameter_state(
    kernel_state,
    output_chart: ChartState,
    input_chart: ChartState,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    _, old_o, old_i = _parameterizations._split(
        kernel_state, output_chart, input_chart, old
    )
    _, new_o, new_i = _parameterizations._split(
        kernel_state, output_chart, input_chart, new
    )
    amplitude_s, state_o, state_i = _parameterizations._split(
        kernel_state, output_chart, input_chart, state
    )
    return torch.cat(
        (
            amplitude_s,
            _profile.transport_state(
                kernel_state.profiles[0], output_chart, old_o, new_o, state_o
            ),
            _profile.transport_state(
                kernel_state.profiles[1], input_chart, old_i, new_i, state_i
            ),
        ),
        dim=-1,
    )
