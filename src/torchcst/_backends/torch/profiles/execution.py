"""Evaluate declared shapes and fixed profile bindings."""

from importlib import import_module

from torchcst.kernels.profiles import (
    BiweightSpec,
    GaussianSpec,
    TriangleSpec,
    TriweightSpec,
    WendlandC2Spec,
)

_TYPES = (GaussianSpec, TriweightSpec, BiweightSpec, TriangleSpec, WendlandC2Spec)


def _module(state):
    if type(state.binding.profile) not in _TYPES:
        raise ValueError("unsupported profile shape or revision")
    if state.binding.profile.revision != 1:
        raise ValueError("unsupported profile shape revision")
    normalization = state.binding.normalization
    if normalization.kind == "discrete_l2":
        expected = None if type(state.binding.profile) is GaussianSpec else 1e-6
        if normalization.floor != expected:
            raise ValueError("unsupported profile normalization floor")
    return import_module(
        "torchcst._backends.torch.profiles."
        + ("gaussian" if type(state.binding.profile) is GaussianSpec else "compact")
    )


def invoke(state, operation, *args, **kwargs):
    return getattr(_module(state), operation)(state, *args, **kwargs)


def parameter_dim(state, chart):
    return chart.center_parameter_dim


def parameter_dof(state, chart):
    return chart.intrinsic_dim


def initialize(state, chart, atoms, *, mode):
    return invoke(state, "initialize", chart, atoms, mode=mode)


def evaluate(state, chart, p):
    return invoke(state, "evaluate", chart, p)


def evaluate_with_precision(state, chart, p, precision):
    return invoke(state, "evaluate_with_precision", chart, p, precision)


def evaluate_with_precision_slice(state, chart, p, precision, selection):
    return invoke(
        state, "evaluate_with_precision_slice", chart, p, precision, selection
    )


def tangent(state, chart, p):
    return invoke(state, "tangent", chart, p)


def tangent_with_precision(state, chart, p, precision):
    return invoke(state, "tangent_with_precision", chart, p, precision)


def project_gradient(state, chart, p, gradient):
    return invoke(state, "project_gradient", chart, p, gradient)


def apply_parameter_update(state, chart, p, displacement):
    return invoke(state, "apply_parameter_update", chart, p, displacement)


def transport_state(state, chart, old, new, vector):
    return invoke(state, "transport_state", chart, old, new, vector)


def shape_function(state, operation, *args):
    module = _module(state)
    name = state.binding.profile.id.replace("wendland_c2", "wendlandc2")
    return getattr(module, name + "_" + operation)(state, *args)
