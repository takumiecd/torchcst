"""PyTorch execution for declared separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.kernels import separable as _kernels
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState


def project_parameter_gradient(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    input_p, output_p = _kernels._split(state, input_chart, output_chart, p)
    input_g, output_g = _kernels._split(state, input_chart, output_chart, gradient)
    return torch.cat(
        (
            _profile.project_gradient(state.profiles[0], input_chart, input_p, input_g),
            _profile.project_gradient(
                state.profiles[1], output_chart, output_p, output_g
            ),
        ),
        dim=-1,
    )


def apply_parameter_update(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    del step_size
    input_p, output_p = _kernels._split(state, input_chart, output_chart, p)
    input_d, output_d = _kernels._split(state, input_chart, output_chart, displacement)
    return torch.cat(
        (
            _profile.apply_parameter_update(
                state.profiles[0], input_chart, input_p, input_d
            ),
            _profile.apply_parameter_update(
                state.profiles[1], output_chart, output_p, output_d
            ),
        ),
        dim=-1,
    )


def transport_parameter_state(
    kernel_state,
    input_chart: ChartState,
    output_chart: ChartState,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    old_i, old_o = _kernels._split(kernel_state, input_chart, output_chart, old)
    new_i, new_o = _kernels._split(kernel_state, input_chart, output_chart, new)
    state_i, state_o = _kernels._split(kernel_state, input_chart, output_chart, state)
    return torch.cat(
        (
            _profile.transport_state(
                kernel_state.profiles[0], input_chart, old_i, new_i, state_i
            ),
            _profile.transport_state(
                kernel_state.profiles[1], output_chart, old_o, new_o, state_o
            ),
        ),
        dim=-1,
    )
