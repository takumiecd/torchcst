"""Exact serialized clock schema for the fixed N1024/B4 research workload.

This CPU evidence validator does not run an optimizer or infer physical alias
identity from flattened snapshots, which clone each serialized path separately.
Five optimizer entries appear under six paths in CST because model extra-state
also exposes the middle optimizer's step Tensor. Other state fields are checked
by the caller's state/gradient/metrics verifier.
"""

import math
import struct

import torch

OPTIMIZER_CLOCK_PATHS = tuple(f"/optimizer/state/{index}/step" for index in range(5))
CST_MODEL_CLOCK_PATH = "/model/2.atom_state.optimizer_state._extra_state/fields/step"
CST_MIDDLE_CLOCK_PATH = "/optimizer/state/4/step"


class ClockSchemaError(ValueError):
    """Clock evidence differs from the frozen research schema."""


def validate_clock_schema(state, *, expected_clock, kind):
    """Validate all serialized step paths and return detached scalar metadata.

    ``state`` is a flat saved CPU Tensor dictionary, not optimizer.state itself.
    Counts describe logical optimizer entries and serialized paths; they do not
    claim five distinct storages were observed in a path-wise cloned snapshot.
    No Tensor values, gradients or version counters are modified or retained.
    """
    if type(kind) is not str or kind not in ("cst", "dense"):
        raise ClockSchemaError("kind must be cst or dense")
    if type(expected_clock) is not int or expected_clock < 0:
        raise ClockSchemaError("expected_clock must be a nonnegative integer")
    try:
        expected_bytes = struct.pack("<f", expected_clock)
        if struct.unpack("<f", expected_bytes)[0] != expected_clock:
            raise ClockSchemaError("expected_clock must be exactly representable in FP32")
    except (OverflowError, struct.error) as error:
        raise ClockSchemaError("expected_clock is outside the FP32 clock schema") from error
    if type(state) is not dict or any(type(key) is not str for key in state):
        raise ClockSchemaError("state must be an exact flat dictionary with string keys")
    expected_paths = set(OPTIMIZER_CLOCK_PATHS)
    if kind == "cst":
        expected_paths.add(CST_MODEL_CLOCK_PATH)
    actual_paths = {key for key in state if key.endswith("/step")}
    if actual_paths != expected_paths:
        raise ClockSchemaError(
            f"clock paths differ: missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    values, encoded = {}, {}
    for path in sorted(expected_paths):
        tensor = state[path]
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.dtype is not torch.float32
            or tensor.layout is not torch.strided
            or tensor.ndim != 0
            or tensor.requires_grad
        ):
            raise ClockSchemaError(f"{path}: expected a detached CPU FP32 scalar Tensor")
        value = torch.Tensor.item(tensor)
        if not math.isfinite(value) or value != expected_clock:
            raise ClockSchemaError(f"{path}: clock differs from {expected_clock}")
        encoded[path] = struct.pack("<f", value)
        if encoded[path] != expected_bytes:
            raise ClockSchemaError(f"{path}: clock bytes differ from the expected scalar")
        values[path] = value
    alias = None
    if kind == "cst":
        if encoded[CST_MODEL_CLOCK_PATH] != encoded[CST_MIDDLE_CLOCK_PATH]:
            raise ClockSchemaError("model clock alias differs from the middle optimizer clock")
        alias = {
            "model_path": CST_MODEL_CLOCK_PATH,
            "optimizer_path": CST_MIDDLE_CLOCK_PATH,
            "value_equal": True,
            "bytes_equal": True,
        }
    return {
        "kind": kind,
        "expected_clock": expected_clock,
        "optimizer_clock_count": 5,
        "unique_optimizer_entry_count": 5,
        "serialized_clock_count": len(expected_paths),
        "clock_dtype": "torch.float32",
        "clock_shape": [],
        "clock_device": "cpu",
        "clock_layout": "torch.strided",
        "optimizer_clock_paths": list(OPTIMIZER_CLOCK_PATHS),
        "serialized_clock_values": values,
        "model_clock_alias": alias,
        "count_scope": "five logical optimizer entries; flattened clone storage identity is not inferred",
    }
