"""Storage-neutral identities and extensible metric declarations."""

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from torchcst._backends.serialization import encode_json


def digest(value):
    return hashlib.sha256(encode_json(value).encode()).hexdigest()


def execution_identity(value):
    """New runs have an execution UUID/UTC time; older artifacts omit both."""
    if "execution_id" not in value and "started_at" not in value:
        return None, None
    try:
        execution_id = value["execution_id"]
        started_at = value["started_at"]
        if type(execution_id) is not str or str(UUID(execution_id)) != execution_id:
            raise ValueError("execution ID must be a canonical UUID")
        if type(started_at) is not str:
            raise ValueError("started_at must be a UTC timestamp")
        stamp = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("started_at must include the UTC offset")
        return execution_id, stamp
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid execution identity/timestamp") from error


@dataclass(frozen=True)
class Metric:
    name: str
    scope: str
    unit: str
    statistic: str
    value: int | float
    samples: list[int | float] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        for text in (self.name, self.scope, self.unit, self.statistic):
            if type(text) is not str or not text or len(text) > 128:
                raise ValueError("metric identifiers need 1..128 characters")
        if type(self.samples) is not list or type(self.details) is not dict:
            raise ValueError("metric samples/details need a list/object")
        for value in [self.value, *self.samples]:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("metric values must be finite and nonnegative")
        encode_json(self.details)


@dataclass(frozen=True)
class Record:
    kind: str
    status: str
    plan_alias: str | None
    environment: dict
    source: dict
    payload: dict
    metrics: tuple[Metric, ...] = ()


@dataclass(frozen=True)
class Projection:
    adapter: str
    revision: int
    case: dict
    plans: dict[str, dict]
    baseline: str
    protocol: dict
    records: tuple[Record, ...]

    @property
    def case_id(self):
        # Display names are not experimental conditions.
        return digest({key: value for key, value in self.case.items() if key != "id"})
