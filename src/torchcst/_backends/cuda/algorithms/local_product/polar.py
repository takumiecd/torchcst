"""Graph-safe specialization of the existing Euclidean polar update."""

import torch

from torchcst._backends.torch.parameterizations import polar_amp_width as polar


def graph_update(value, p, displacement, *, step_size):
    """Production polar update algebra, specialized to fixed Euclidean centers.

    Avoids public geometry's host finite-value checks during CUDA Graph capture.
    Tests compare against execution.apply_parameter_update; no width SGD.
    """
    from torchcst._backends.torch.updates.polar_amp_width import _project_polar

    polar = _project_polar(p[:, :2])
    q = polar.square().sum(-1, keepdim=True)
    raw = displacement[:, :2]
    tangent = raw - ((raw * polar).sum(-1, keepdim=True) / q) * polar
    chord = polar + tangent
    direction = chord / chord.square().sum(-1, keepdim=True).sqrt()
    energy = tangent.square().sum(-1, keepdim=True)
    if value.setting("activity_mode") == "time_energy":
        energy = energy / step_size
    amp, _ = polar_module_amplitude(value, direction)
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


polar_module_amplitude = polar._amplitude_and_alpha
