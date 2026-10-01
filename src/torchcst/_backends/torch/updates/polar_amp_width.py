"""PyTorch execution for legacy polar_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst.geometry import Chart


def apply_parameter_update(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    """Project task motion tangentially, decay radius, then enforce 1<=q<=4."""

    self._split(input_chart, output_chart, p)
    if displacement.shape != p.shape:
        raise ValueError("displacement must match the atom parameter shape")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")

    polar = self._project_polar(p[:, :2])
    raw = displacement[:, :2]
    radius_square = polar.square().sum(dim=-1, keepdim=True)
    radial_coefficient = (raw * polar).sum(dim=-1, keepdim=True) / radius_square
    tangent = raw - radial_coefficient * polar

    # Preserve the optimizer's angular proposal while allowing the radial
    # history clock to be calibrated independently. ``finite_chord`` is
    # the original q' = q + gamma ||tangent||^2 rule. ``time_energy``
    # interprets gamma as an activity rate and divides the squared motion
    # by the optimizer's outer time step.
    chord = polar + tangent
    chord_q = chord.square().sum(dim=-1, keepdim=True)
    direction = chord / chord_q.sqrt()
    energy = tangent.square().sum(dim=-1, keepdim=True)
    if self.activity_mode == "time_energy":
        energy = energy / step_size
    proposed_amplitude, _ = self._amplitude_and_alpha(direction)
    dormant_weight = 1.0 / (1.0 + (proposed_amplitude / self.w_c.to(p)).square())
    dormant_q = (
        3.0
        * self.dormant_expansion_rate.to(p)
        * p.new_tensor(step_size)
        * dormant_weight.unsqueeze(-1)
    )
    task_q = (radius_square + self.activity_gain.to(p) * energy + dormant_q).clamp(
        1.0, 4.0
    )
    task_polar = direction * task_q.sqrt()

    # Exact gradient flow for R(q)=lambda/2*(q-1)^2 over time step_size:
    # y=(q-1)/q decays as exp(-4*lambda*t), keeping q in [1, 4].
    decay = torch.exp(-4.0 * self.radial_regularization.to(p) * p.new_tensor(step_size))
    activity = (task_q - 1.0) / task_q
    regularized_q = 1.0 / (1.0 - activity * decay)
    regularized_polar = task_polar * (regularized_q / task_q).sqrt()

    _, input_p, output_p = self._split(input_chart, output_chart, p)
    _, input_d, output_d = self._split(
        input_chart,
        output_chart,
        displacement,
    )
    updated_input = self.profile.apply_parameter_update(
        input_chart,
        input_p,
        input_d,
    )
    updated_output = self.profile.apply_parameter_update(
        output_chart,
        output_p,
        output_d,
    )
    return torch.cat((regularized_polar, updated_input, updated_output), dim=-1)


def project_parameter_gradient(
    self,
    input_chart: Chart,
    output_chart: Chart,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    _, input_p, output_p = self._split(input_chart, output_chart, p)
    polar_g, input_g, output_g = self._split(
        input_chart,
        output_chart,
        gradient,
    )
    return torch.cat(
        (
            polar_g,
            self.profile.project_gradient(input_chart, input_p, input_g),
            self.profile.project_gradient(output_chart, output_p, output_g),
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
    _, old_i, old_o = self._split(input_chart, output_chart, old)
    _, new_i, new_o = self._split(input_chart, output_chart, new)
    polar_s, state_i, state_o = self._split(input_chart, output_chart, state)
    return torch.cat(
        (
            polar_s,
            self.profile.transport_state(input_chart, old_i, new_i, state_i),
            self.profile.transport_state(output_chart, old_o, new_o, state_o),
        ),
        dim=-1,
    )


def _project_polar(polar: Tensor) -> Tensor:
    radius_square = polar.square().sum(dim=-1, keepdim=True)
    tiny = torch.finfo(polar.dtype).tiny
    safe_radius = radius_square.clamp_min(tiny).sqrt()
    fallback = torch.zeros_like(polar)
    fallback[:, 1] = 1.0
    unit = torch.where(radius_square > tiny, polar / safe_radius, fallback)
    radius = safe_radius.clamp(1.0, 2.0)
    return unit * radius
