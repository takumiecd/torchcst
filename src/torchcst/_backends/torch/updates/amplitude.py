"""Amplitude proposals composed with the inner coordinate policy."""

import torch

from ..kernels import amplitude as _amplitude
from ..kernels import execution as _kernel


def project_parameter_gradient(state, *charts_p_gradient):
    *charts, p, gradient = charts_p_gradient
    _, inner = _amplitude._split(state, *charts, p)
    head, tail = _amplitude._split(state, *charts, gradient)
    return torch.cat(
        (head, _kernel.project_parameter_gradient(state.inner, *charts, inner, tail)),
        -1,
    )


def apply_parameter_update(state, *charts_p_displacement, step_size):
    *charts, p, displacement = charts_p_displacement
    head, inner = _amplitude._split(state, *charts, p)
    head_delta, tail_delta = _amplitude._split(state, *charts, displacement)
    return torch.cat(
        (
            head + head_delta,
            _kernel.apply_parameter_update(
                state.inner, *charts, inner, tail_delta, step_size=step_size
            ),
        ),
        -1,
    )


def transport_parameter_state(state, *charts_old_new_vector):
    *charts, old, new, vector = charts_old_new_vector
    _, old_inner = _amplitude._split(state, *charts, old)
    _, new_inner = _amplitude._split(state, *charts, new)
    head, tail = _amplitude._split(state, *charts, vector)
    return torch.cat(
        (
            head,
            _kernel.transport_parameter_state(
                state.inner, *charts, old_inner, new_inner, tail
            ),
        ),
        -1,
    )
