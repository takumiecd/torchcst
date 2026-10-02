"""Structural exact conditions, with no tensor values or executable decoding."""

import math
from dataclasses import fields, is_dataclass
from functools import lru_cache

import torch


def _declaration(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                f.name: _declaration(getattr(value, f.name)) for f in fields(value)
            },
        }
    if type(value) is tuple:
        return [_declaration(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("operator conditions require immutable scalar declarations")


def _freeze(value):
    # Distinguish bool from numbers (True == 1 in Python). Numeric declarations
    # such as origin=0 and origin=0.0 describe the same coordinates.
    if type(value) is dict:
        return ("object", tuple((k, _freeze(v)) for k, v in sorted(value.items())))
    if type(value) is list:
        return ("array", tuple(_freeze(v) for v in value))
    if type(value) in (int, float):
        return ("number", value)
    return (type(value).__name__, value)


@lru_cache(maxsize=128)
def _operator_key(operator):
    return _freeze(_declaration(operator))


def _metadata(context):
    return (
        context.input_shape,
        context.input_strides,
        str(context.dtype),
        context.atom_count,
        context.parameter_dim,
        context.device.type,
        context.device.name,
        context.device.compute_capability,
        context.device.sm_count,
        context.required_grads.inputs,
        context.required_grads.parameters,
        context.execution_mode,
        context.deterministic,
        context.precision.autocast,
        context.precision.allow_tf32,
        context.workspace_limit_bytes,
    )


def context_key(context):
    """Hashable metadata; no JSON, file I/O, atom inspection or device queries."""
    return (_operator_key(context.operator), _metadata(context))


def dump_condition(context):
    """Export readable conditions; device index is intentionally not portable."""
    return {
        "operator": _declaration(context.operator),
        "input_shape": list(context.input_shape),
        "input_strides": list(context.input_strides),
        "dtype": str(context.dtype),
        "atom_count": context.atom_count,
        "parameter_dim": context.parameter_dim,
        "device": {
            "type": context.device.type,
            "name": context.device.name,
            "compute_capability": (
                list(context.device.compute_capability)
                if context.device.compute_capability is not None
                else None
            ),
            "sm_count": context.device.sm_count,
        },
        "required_grads": {
            "inputs": context.required_grads.inputs,
            "parameters": context.required_grads.parameters,
        },
        "execution_mode": context.execution_mode,
        "deterministic": context.deterministic,
        "precision": {
            "autocast": context.precision.autocast,
            "allow_tf32": context.precision.allow_tf32,
        },
        "workspace_limit_bytes": context.workspace_limit_bytes,
    }


def exact_fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} needs exactly: {', '.join(sorted(keys))}")


def _integer(value, label, minimum=0, nullable=False):
    if nullable and value is None:
        return
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _declaration_data(value):
    if type(value) is dict:
        exact_fields(value, ("type", "fields"), "declaration")
        if type(value["type"]) is not str or not value["type"]:
            raise ValueError("declaration type must be a nonempty string")
        if type(value["fields"]) is not dict or not all(
            type(k) is str for k in value["fields"]
        ):
            raise ValueError("declaration fields must be a string-keyed object")
        for item in value["fields"].values():
            _declaration_data(item)
    elif type(value) is list:
        for item in value:
            _declaration_data(item)
    elif value is None or type(value) in (str, bool, int):
        return
    elif type(value) is float and math.isfinite(value):
        return
    else:
        raise ValueError("invalid declaration value")


def condition_key(data):
    """Decode keys only; declaration type strings never import or execute code."""
    exact_fields(
        data,
        (
            "operator",
            "input_shape",
            "input_strides",
            "dtype",
            "atom_count",
            "parameter_dim",
            "device",
            "required_grads",
            "execution_mode",
            "deterministic",
            "precision",
            "workspace_limit_bytes",
        ),
        "condition",
    )
    _declaration_data(data["operator"])
    if type(data["operator"]) is not dict:
        raise ValueError("operator must be a typed declaration")
    shape, strides = data["input_shape"], data["input_strides"]
    if (
        type(shape) is not list
        or not shape
        or type(strides) is not list
        or len(shape) != len(strides)
    ):
        raise ValueError("shape and strides must be matching arrays")
    for n in (*shape, *strides):
        _integer(n, "shape/stride")
    dtype = data["dtype"]
    if (
        type(dtype) is not str
        or not dtype.startswith("torch.")
        or not isinstance(
            getattr(torch, dtype.removeprefix("torch."), None), torch.dtype
        )
    ):
        raise ValueError("unknown condition dtype")
    _integer(data["atom_count"], "atom_count")
    _integer(data["parameter_dim"], "parameter_dim", 1)
    _integer(data["workspace_limit_bytes"], "workspace_limit_bytes", nullable=True)
    device = data["device"]
    exact_fields(device, ("type", "name", "compute_capability", "sm_count"), "device")
    if (
        type(device["type"]) is not str
        or not device["type"]
        or type(device["name"]) is not str
    ):
        raise ValueError("device type/name must be strings")
    cc = device["compute_capability"]
    if cc is not None:
        if type(cc) is not list or len(cc) != 2:
            raise ValueError("compute_capability must have two integers")
        for n in cc:
            _integer(n, "compute_capability")
    _integer(device["sm_count"], "sm_count", 1, nullable=True)
    grads, precision = data["required_grads"], data["precision"]
    exact_fields(grads, ("inputs", "parameters"), "required_grads")
    exact_fields(precision, ("autocast", "allow_tf32"), "precision")
    if any(
        type(v) is not bool
        for v in (*grads.values(), *precision.values(), data["deterministic"])
    ):
        raise ValueError("gradient, precision and deterministic flags must be bool")
    if data["execution_mode"] not in ("eager", "cuda_graph"):
        raise ValueError("unknown execution_mode")
    return (
        _freeze(data["operator"]),
        (
            tuple(shape),
            tuple(strides),
            dtype,
            data["atom_count"],
            data["parameter_dim"],
            device["type"],
            device["name"],
            tuple(cc) if cc is not None else None,
            device["sm_count"],
            grads["inputs"],
            grads["parameters"],
            data["execution_mode"],
            data["deterministic"],
            precision["autocast"],
            precision["allow_tf32"],
            data["workspace_limit_bytes"],
        ),
    )
