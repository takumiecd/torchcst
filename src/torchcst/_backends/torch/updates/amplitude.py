"""PyTorch execution for legacy amplitude contracts."""

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
    _, inner = self._split(input_chart, output_chart, p)
    amplitude_gradient, inner_gradient = self._split(
        input_chart, output_chart, gradient
    )
    projected = self.kernel.project_parameter_gradient(
        input_chart,
        output_chart,
        inner,
        inner_gradient,
    )
    return torch.cat((amplitude_gradient, projected), dim=-1)


def apply_parameter_update(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    amplitude, inner = self._split(input_chart, output_chart, p)
    amplitude_delta, inner_delta = self._split(input_chart, output_chart, displacement)
    updated = self.kernel.apply_parameter_update(
        input_chart,
        output_chart,
        inner,
        inner_delta,
        step_size=step_size,
    )
    return torch.cat((amplitude + amplitude_delta, updated), dim=-1)


def transport_parameter_state(
    self,
    input_chart: Chart,
    output_chart: Chart,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    _, old_inner = self._split(input_chart, output_chart, old)
    _, new_inner = self._split(input_chart, output_chart, new)
    amplitude_state, inner_state = self._split(input_chart, output_chart, state)
    transported = self.kernel.transport_parameter_state(
        input_chart,
        output_chart,
        old_inner,
        new_inner,
        inner_state,
    )
    return torch.cat((amplitude_state, transported), dim=-1)
