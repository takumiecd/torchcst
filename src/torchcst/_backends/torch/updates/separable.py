"""PyTorch execution for legacy separable contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart


def project_parameter_gradient(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    input_p, output_p = self._split(input_chart, output_chart, p)
    input_g, output_g = self._split(input_chart, output_chart, gradient)
    return torch.cat(
        (
            self.input_profile.project_gradient(input_chart, input_p, input_g),
            self.output_profile.project_gradient(output_chart, output_p, output_g),
        ),
        dim=-1,
    )


def apply_parameter_update(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    del step_size
    input_p, output_p = self._split(input_chart, output_chart, p)
    input_d, output_d = self._split(input_chart, output_chart, displacement)
    return torch.cat(
        (
            self.input_profile.apply_parameter_update(input_chart, input_p, input_d),
            self.output_profile.apply_parameter_update(
                output_chart, output_p, output_d
            ),
        ),
        dim=-1,
    )


def transport_parameter_state(
    self,
    input_chart: Chart,
    output_chart: Chart,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    old_i, old_o = self._split(input_chart, output_chart, old)
    new_i, new_o = self._split(input_chart, output_chart, new)
    state_i, state_o = self._split(input_chart, output_chart, state)
    return torch.cat(
        (
            self.input_profile.transport_state(input_chart, old_i, new_i, state_i),
            self.output_profile.transport_state(output_chart, old_o, new_o, state_o),
        ),
        dim=-1,
    )
