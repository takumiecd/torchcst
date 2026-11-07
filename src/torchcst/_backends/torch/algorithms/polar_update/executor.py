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
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
    displacement: Tensor,
    *,
    step_size: float,
) -> Tensor:
    """Project task motion tangentially, decay radius, then enforce 1<=q<=4."""

    _parameterizations._split(state, output_chart, input_chart, p)
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
    center_update = None
    if state.setting("activity_mode") == "center_chord":
        center_update, energy = _center_chord(
            state, output_chart, input_chart, p, displacement
        )
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

    if center_update is None:
        _, output_p, input_p = _parameterizations._split(
            state, output_chart, input_chart, p
        )
        _, output_d, input_d = _parameterizations._split(
            state, output_chart, input_chart, displacement
        )
        updated_output = _profile.apply_parameter_update(
            state.profiles[0], output_chart, output_p, output_d
        )
        updated_input = _profile.apply_parameter_update(
            state.profiles[-1], input_chart, input_p, input_d
        )
    else:
        updated_output, updated_input = center_update
    return torch.cat((regularized_polar, updated_output, updated_input), dim=-1)


def _center_chord(state, output_chart, input_chart, p, displacement):
    """Accepted shortest torus motion / old physical support, mean over axes.

    Only center motion supplies activity. Amplitude keeps its original angular
    proposal, and the existing radial decay and optional dormant term remain.
    Old support is frozen before this update; no circular radius equation or
    hidden optimizer state is introduced.
    """
    from torchcst.geometry.spec import FlatTorusGeometrySpec
    from torchcst.kernels.parameterizations import PolarAmpRadiusSpec

    if type(state.spec.parameterization) is not PolarAmpRadiusSpec or any(
        type(chart.geometry.spec) is not FlatTorusGeometrySpec
        or chart.geometry.spec.intrinsic_dim != 1
        for chart in (output_chart, input_chart)
    ):
        raise ValueError(
            "center_chord requires radius Polar and scalar FlatTorus centers"
        )
    polar, old_o, old_i = _parameterizations._split(state, output_chart, input_chart, p)
    _, delta_o, delta_i = _parameterizations._split(
        state, output_chart, input_chart, displacement
    )
    new_o = _profile.apply_parameter_update(
        state.profiles[0], output_chart, old_o, delta_o
    )
    new_i = _profile.apply_parameter_update(
        state.profiles[-1], input_chart, old_i, delta_i
    )
    a, alpha = _parameterizations._amplitude_and_alpha(state, _project_polar(polar))
    names = state.spec.parameterization.width_scalar_names
    ri, _, _ = _parameterizations._width_bounds(
        state, a, alpha, side="input", scalar_names=names
    )
    ro, _, _ = _parameterizations._width_bounds(
        state, a, alpha, side="output", scalar_names=names
    )
    energies = []
    for old, new, radius, chart in (
        (old_o, new_o, ro, output_chart),
        (old_i, new_i, ri, input_chart),
    ):
        period = p.new_tensor(chart.geometry.spec.periods)
        delta = new - old
        delta = delta - period * torch.floor(delta / period + 0.5)
        energies.append((delta / radius[:, None]).square().sum(-1, keepdim=True))
    return (new_o, new_i), (energies[0] + energies[1]) * 0.5


def project_parameter_gradient(
    state,
    output_chart: ChartState,
    input_chart: ChartState,
    p: Tensor,
    gradient: Tensor,
) -> Tensor:
    (_, output_p, input_p), (polar_g, output_g, input_g) = (
        _parameterizations._split_many(state, output_chart, input_chart, p, gradient)
    )
    projected_output = _profile.project_gradient(
        state.profiles[0], output_chart, output_p, output_g
    )
    projected_input = _profile.project_gradient(
        state.profiles[-1], input_chart, input_p, input_g
    )
    # Geometry callbacks still run their validation. Euclidean/FlatTorus
    # callbacks return the original views, so no reassembly is needed.
    if projected_output is output_g and projected_input is input_g:
        return gradient
    return torch.cat(
        (polar_g, projected_output, projected_input),
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
    (_, old_o, old_i), (_, new_o, new_i), (polar_s, state_o, state_i) = (
        _parameterizations._split_many(
            kernel_state, output_chart, input_chart, old, new, state
        )
    )
    transported_output = _profile.transport_state(
        kernel_state.profiles[0], output_chart, old_o, new_o, state_o
    )
    transported_input = _profile.transport_state(
        kernel_state.profiles[-1], input_chart, old_i, new_i, state_i
    )
    if transported_output is state_o and transported_input is state_i:
        return state
    return torch.cat(
        (polar_s, transported_output, transported_input),
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

    if value.setting("activity_mode") == "center_chord":
        raise ValueError("center_chord requires eager scalar FlatTorus execution")
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
