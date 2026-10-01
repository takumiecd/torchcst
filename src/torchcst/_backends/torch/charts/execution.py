"""Functional chart evaluation without configuration snapshots."""

from importlib import import_module

from torchcst.geometry.state import ChartState


def _evaluate(state, operation, *args, **kwargs):
    if type(state) is not ChartState or state.spec.revision != 1:
        raise ValueError("unsupported chart state or revision")
    module = import_module(f"torchcst._backends.torch.charts.{state.spec.kind}")
    function = getattr(module, operation, None)
    if function is None and state.spec.kind != "explicit":
        from . import lazy

        function = getattr(lazy, operation, None)
    if function is None:
        raise NotImplementedError(f"{operation} is undefined for {state.spec.kind}")
    return function(state, *args, **kwargs)


def positions(state, *args, **kwargs):
    return _evaluate(state, "positions", *args, **kwargs)


def squared_distance(state, *args, **kwargs):
    return _evaluate(state, "squared_distance", *args, **kwargs)


def center_offsets(state, *args, **kwargs):
    return _evaluate(state, "center_offsets", *args, **kwargs)


def initialize_centers(state, *args, **kwargs):
    return _evaluate(state, "initialize_centers", *args, **kwargs)


def tile_indices(state, *args, **kwargs):
    return _evaluate(state, "tile_indices", *args, **kwargs)


def validate_support(state, *args, **kwargs):
    return _evaluate(state, "validate_support", *args, **kwargs)


def _bounds(state, *args, **kwargs):
    return _evaluate(state, "_bounds", *args, **kwargs)
