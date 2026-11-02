"""Independent geometry updates for fixed-width radial centers."""

import torch

from torchcst._backends.torch.profiles import execution as _profile


def _apply(state, charts, tables, operation):
    parts = []
    start = 0
    for chart in charts:
        end = start + chart.center_parameter_dim
        parts.append(
            operation(state.profiles[0], chart, *(t[:, start:end] for t in tables))
        )
        start = end
    return torch.cat(parts, dim=-1)


def project_parameter_gradient(state, *charts_p_gradient):
    *charts, p, gradient = charts_p_gradient
    return _apply(state, charts, (p, gradient), _profile.project_gradient)


def apply_parameter_update(state, *charts_p_displacement, step_size):
    *charts, p, displacement = charts_p_displacement
    return _apply(state, charts, (p, displacement), _profile.apply_parameter_update)


def transport_parameter_state(state, *charts_old_new_vector):
    *charts, old, new, vector = charts_old_new_vector
    return _apply(state, charts, (old, new, vector), _profile.transport_state)
