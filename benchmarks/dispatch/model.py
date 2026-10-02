"""Generic per-run aggregates supplied to trusted, user-defined score functions."""

import math
import statistics
from dataclasses import dataclass

from torchcst._backends.cuda.schema import DispatchContext, ExecutionPlan


@dataclass(frozen=True, order=True)
class MetricKey:
    name: str
    scope: str
    unit: str
    statistic: str


@dataclass(frozen=True)
class Summary:
    count: int
    median: float
    mean: float
    stdev: float | None
    variance: float | None
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values):
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("aggregate needs finite nonnegative observations")
        return cls(
            len(values),
            statistics.median(values),
            statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else None,
            statistics.variance(values) if len(values) > 1 else None,
            min(values),
            max(values),
        )


@dataclass(frozen=True)
class Candidate:
    context: DispatchContext
    plan: ExecutionPlan
    plan_id: str
    cohort: dict
    observations: tuple[dict, ...]
    metrics: dict[MetricKey, Summary]

    @property
    def run_count(self):
        return len(self.observations)

    def metric(self, name, scope, unit, statistic):
        return self.metrics.get(MetricKey(name, scope, unit, statistic))
