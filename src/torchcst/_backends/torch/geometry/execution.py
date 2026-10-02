"""Select geometry evaluation from an immutable contract and live Tensor state."""

from importlib import import_module

from torchcst.geometry.spec import (
    EuclideanGeometrySpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)
from torchcst.geometry.state import GeometryState


def _evaluate(state, operation, *args, **kwargs):
    if type(state) is not GeometryState or state.spec.revision != 1:
        raise ValueError("unsupported geometry state or revision")
    family = {
        EuclideanGeometrySpec: "euclidean",
        SphereGeometrySpec: "sphere",
        TorusGeometrySpec: "torus",
    }.get(type(state.spec))
    if family is None:
        raise ValueError("unsupported geometry declaration")
    module = import_module(f"torchcst._backends.torch.geometry.{family}")
    function = getattr(module, operation, None)
    if function is None:
        from . import base

        function = getattr(base, operation, None)
    if function is None:
        raise NotImplementedError(f"{operation} is undefined for {family}")
    return function(state, *args, **kwargs)


def validate_points(state, *args, **kwargs):
    return _evaluate(state, "validate_points", *args, **kwargs)


def validate_centers(state, *args, **kwargs):
    return _evaluate(state, "validate_centers", *args, **kwargs)


def decode_centers(state, *args, **kwargs):
    return _evaluate(state, "decode_centers", *args, **kwargs)


def encode_centers(state, *args, **kwargs):
    return _evaluate(state, "encode_centers", *args, **kwargs)


def lift_chart_coordinates(state, *args, **kwargs):
    return _evaluate(state, "lift_chart_coordinates", *args, **kwargs)


def squared_distance(state, *args, **kwargs):
    return _evaluate(state, "squared_distance", *args, **kwargs)


def center_offsets(state, *args, **kwargs):
    return _evaluate(state, "center_offsets", *args, **kwargs)


def initialize_centers(state, *args, **kwargs):
    return _evaluate(state, "initialize_centers", *args, **kwargs)


def project_tangent(state, *args, **kwargs):
    return _evaluate(state, "project_tangent", *args, **kwargs)


def retract(state, *args, **kwargs):
    return _evaluate(state, "retract", *args, **kwargs)


def transport(state, *args, **kwargs):
    return _evaluate(state, "transport", *args, **kwargs)


def sample_sites(state, *args, **kwargs):
    return _evaluate(state, "sample_sites", *args, **kwargs)


def lift_tangent_sites(state, *args, **kwargs):
    return _evaluate(state, "lift_tangent_sites", *args, **kwargs)


def max_parameter_radius(state, *args, **kwargs):
    return _evaluate(state, "max_parameter_radius", *args, **kwargs)


def max_section_parameter_radius(state, *args, **kwargs):
    return _evaluate(state, "max_section_parameter_radius", *args, **kwargs)


def circumference(state, *args, **kwargs):
    return _evaluate(state, "circumference", *args, **kwargs)


def axis_separation_lower_bound(state, *args, **kwargs):
    return _evaluate(state, "axis_separation_lower_bound", *args, **kwargs)
