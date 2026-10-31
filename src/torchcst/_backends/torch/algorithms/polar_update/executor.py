"""PyTorch execution for declared polar_amplitude_bandwidth contracts."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchcst._backends.torch.parameterizations import (
    polar_amp_width as _parameterizations,
)
from torchcst._backends.torch.profiles import execution as _profile
from torchcst._builtin_callbacks import seal_functions as _seal_functions
from torchcst.charts import ChartState


def decay_amplitude(previous: Tensor, updated: Tensor, *, amount: float) -> Tensor:
    """Subtract decoupled old-amplitude decay at fixed accepted Polar activity.

    AdamW's decay uses the old value, not the task increment. The bounded
    amplitude is the normalized first Polar coordinate. Reconstruct the pair
    on its accepted circle, retaining the second coordinate's signed branch.
    Centers and Adam state are not decayed. Zero decay keeps the old path exact.
    """
    if amount == 0:
        return updated
    old_pair, new_pair = previous[:, :2], updated[:, :2]
    tiny = torch.finfo(previous.dtype).tiny
    old_radius = old_pair.square().sum(-1).clamp_min(tiny).sqrt()
    new_radius = new_pair.square().sum(-1).clamp_min(tiny).sqrt()
    old_amplitude = old_pair[:, 0] / old_radius
    target = (new_pair[:, 0] / new_radius - amount * old_amplitude).clamp(-1, 1)
    # (1-|u|)(1+|u|) avoids cancellation near the bounded endpoints.
    absolute = target.abs()
    magnitude = new_radius * ((1 - absolute) * (1 + absolute)).clamp_min(0).sqrt()
    s = new_radius * target
    t = torch.copysign(magnitude, new_pair[:, 1])
    return torch.cat((s[:, None], t[:, None], updated[:, 2:]), dim=-1)


def apply_parameter_update(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    """Project task motion tangentially, decay radius, then enforce 1<=q<=4."""

    _parameterizations._split(state, input_chart, output_chart, p)
    if displacement.shape != p.shape:
        raise ValueError("displacement must match the atom parameter shape")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")

    polar = _project_polar(p[:, :2])
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
    if state.setting("activity_mode") == "time_energy":
        energy = energy / step_size
    proposed_amplitude, _ = _parameterizations._amplitude_and_alpha(state, direction)
    dormant_weight = 1.0 / (
        1.0 + (proposed_amplitude / state.scalar("w_c").to(p)).square()
    )
    dormant_q = (
        3.0
        * state.scalar("dormant_expansion_rate").to(p)
        * p.new_tensor(step_size)
        * dormant_weight.unsqueeze(-1)
    )
    task_q = (
        radius_square + state.scalar("activity_gain").to(p) * energy + dormant_q
    ).clamp(1.0, 4.0)
    task_polar = direction * task_q.sqrt()

    # Exact gradient flow for R(q)=lambda/2*(q-1)^2 over time step_size:
    # y=(q-1)/q decays as exp(-4*lambda*t), keeping q in [1, 4].
    decay = torch.exp(
        -4.0 * state.scalar("radial_regularization").to(p) * p.new_tensor(step_size)
    )
    activity = (task_q - 1.0) / task_q
    regularized_q = 1.0 / (1.0 - activity * decay)
    regularized_polar = task_polar * (regularized_q / task_q).sqrt()

    _, input_p, output_p = _parameterizations._split(
        state, input_chart, output_chart, p
    )
    _, input_d, output_d = _parameterizations._split(
        state, input_chart, output_chart, displacement
    )
    updated_input = _profile.apply_parameter_update(
        state.profiles[0], input_chart, input_p, input_d
    )
    updated_output = _profile.apply_parameter_update(
        state.profiles[1], output_chart, output_p, output_d
    )
    return torch.cat((regularized_polar, updated_input, updated_output), dim=-1)


def project_parameter_gradient(
    state,
    input_chart: ChartState,
    output_chart: ChartState,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    (_, input_p, output_p), (polar_g, input_g, output_g) = (
        _parameterizations._split_many(state, input_chart, output_chart, p, gradient)
    )
    projected_input = _profile.project_gradient(
        state.profiles[0], input_chart, input_p, input_g
    )
    projected_output = _profile.project_gradient(
        state.profiles[1], output_chart, output_p, output_g
    )
    # Geometry callbacks still run their validation. Euclidean/FlatTorus
    # callbacks return the original views, so no reassembly is needed.
    if projected_input is input_g and projected_output is output_g:
        return gradient
    return torch.cat(
        (polar_g, projected_input, projected_output),
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
    (_, old_i, old_o), (_, new_i, new_o), (polar_s, state_i, state_o) = (
        _parameterizations._split_many(
            kernel_state, input_chart, output_chart, old, new, state
        )
    )
    transported_input = _profile.transport_state(
        kernel_state.profiles[0], input_chart, old_i, new_i, state_i
    )
    transported_output = _profile.transport_state(
        kernel_state.profiles[1], output_chart, old_o, new_o, state_o
    )
    if transported_input is state_i and transported_output is state_o:
        return state
    return torch.cat(
        (polar_s, transported_input, transported_output),
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


def graph_update(value, p, displacement, *, step_size):
    """Production polar update algebra, specialized to fixed Euclidean centers.

    Avoids public geometry's host finite-value checks during CUDA Graph capture.
    Tests compare against execution.apply_parameter_update; no width SGD.
    """

    polar = _project_polar(p[:, :2])
    q = polar.square().sum(-1, keepdim=True)
    raw = displacement[:, :2]
    tangent = raw - ((raw * polar).sum(-1, keepdim=True) / q) * polar
    chord = polar + tangent
    direction = chord / chord.square().sum(-1, keepdim=True).sqrt()
    energy = tangent.square().sum(-1, keepdim=True)
    if value.setting("activity_mode") == "time_energy":
        energy = energy / step_size
    amp, _ = _parameterizations._amplitude_and_alpha(value, direction)
    dormant = 1 / (1 + (amp / value.scalar("w_c").to(p)).square())
    task_q = (
        q
        + value.scalar("activity_gain").to(p) * energy
        + 3
        * value.scalar("dormant_expansion_rate").to(p)
        * step_size
        * dormant[:, None]
    ).clamp(1.0, 4.0)
    decay = torch.exp(-4 * value.scalar("radial_regularization").to(p) * step_size)
    regularized = 1 / (1 - (task_q - 1) / task_q * decay)
    # Preserve the same operation order as production, including radius rescaling.
    updated = direction * task_q.sqrt() * (regularized / task_q).sqrt()
    return torch.cat((updated, p[:, 2:] + displacement[:, 2:]), 1)


# Optional prepared updates may specialize only these original definitions.

_seal_functions(globals())
