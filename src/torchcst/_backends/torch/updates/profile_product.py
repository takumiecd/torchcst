"""Reuse the Polar law with one geometry-owned center vector per atom."""

import math

import torch

from torchcst._backends.torch.algorithms.polar_update import executor as _polar_update
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.parameterizations import profile_product as _coordinates


def apply_parameter_update(state, chart, p, displacement, *, step_size):
    _, centers = _coordinates._split(state, chart, p)
    if displacement.shape != p.shape:
        raise ValueError("displacement must match the atom parameter shape")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    updated = _polar_update.graph_update(state, p, displacement, step_size=step_size)
    # Retain public geometry validation for the eager reference.
    new_centers = _geometry.retract(chart.geometry, centers, displacement[:, 2:])
    return torch.cat((updated[:, :2], new_centers), dim=-1)


def project_parameter_gradient(state, chart, p, gradient):
    _, centers = _coordinates._split(state, chart, p)
    _coordinates._split(state, chart, gradient)
    center_g = _geometry.project_tangent(chart.geometry, centers, gradient[:, 2:])
    return torch.cat((gradient[:, :2], center_g), dim=-1)


def transport_parameter_state(state, chart, old, new, vector):
    _, old_centers = _coordinates._split(state, chart, old)
    _, new_centers = _coordinates._split(state, chart, new)
    _coordinates._split(state, chart, vector)
    center_s = _geometry.transport(
        chart.geometry, old_centers, new_centers, vector[:, 2:]
    )
    return torch.cat((vector[:, :2], center_s), dim=-1)


def _project_polar(polar):
    return _polar_update._project_polar(polar)
