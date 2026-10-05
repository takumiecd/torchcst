"""Strict JSON for declarative metadata; no tensor or execution-code encoding."""

from __future__ import annotations

import json
import math


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number: {value}")
    return number


def decode_json(value: str | bytes | bytearray):
    """Reject duplicate fields, nonfinite literals and overflowing numbers."""
    return json.loads(
        value, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float
    )


def _validate(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON metadata requires finite numbers")
        return
    if type(value) is list:
        for item in value:
            _validate(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON metadata requires string keys")
            _validate(item)
        return
    raise TypeError("JSON metadata requires scalars, lists and string-keyed dicts")


def encode_json(value) -> str:
    """Encode JSON-native metadata deterministically, without lossy coercions."""
    _validate(value)
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
