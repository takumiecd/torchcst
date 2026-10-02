"""Geometry updates for fixed-width radial coordinates."""

from torchcst._backends.torch.profiles import execution as _profile


def project_parameter_gradient(state, chart, p, gradient):
    return _profile.project_gradient(state.profiles[0], chart, p, gradient)


def apply_parameter_update(state, chart, p, displacement, *, step_size):
    return _profile.apply_parameter_update(state.profiles[0], chart, p, displacement)


def transport_parameter_state(state, chart, old, new, vector):
    return _profile.transport_state(state.profiles[0], chart, old, new, vector)
