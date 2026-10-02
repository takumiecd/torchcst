"""Euclidean updates of signed amplitude, log width and centers."""


def project_parameter_gradient(state, chart, p, gradient):
    return gradient


def apply_parameter_update(state, chart, p, displacement, *, step_size):
    return p + displacement


def transport_parameter_state(state, chart, old, new, vector):
    return vector
