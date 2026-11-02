"""PyTorch execution for declared separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import separable as _kernels
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import ChartState


def project_parameter_gradient(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    output_p, input_p = _kernels._split(state, output_chart, input_chart, p)
    output_g, input_g = _kernels._split(state, output_chart, input_chart, gradient)
    return torch.cat(
        (
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
    output_p, input_p = _kernels._split(state, output_chart, input_chart, p)
    output_d, input_d = _kernels._split(state, output_chart, input_chart, displacement)
    return torch.cat(
        (
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
    old_o, old_i = _kernels._split(kernel_state, output_chart, input_chart, old)
    new_o, new_i = _kernels._split(kernel_state, output_chart, input_chart, new)
    state_o, state_i = _kernels._split(kernel_state, output_chart, input_chart, state)
    return torch.cat(
        (
            _profile.transport_state(
                kernel_state.profiles[0], output_chart, old_o, new_o, state_o
            ),
            _profile.transport_state(
                kernel_state.profiles[1], input_chart, old_i, new_i, state_i
            ),
        ),
        dim=-1,
    )
