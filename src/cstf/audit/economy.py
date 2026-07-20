"""Aggregate-only lifecycle economy audit."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from cstf.storage import (
    NeuronRetire,
    NeuronUngate,
    SynapseBirth,
    SynapseDeath,
)


def _tensor_copy(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("audit tensor values must be Tensors")
    return value.detach().to(device="cpu").clone()


@dataclass(frozen=True)
class AuditRecord:
    """One read-only event snapshot pushed by the engine.

    ``prune_ages`` and ``rent_thresholds`` are the values observed immediately
    before adjudication. ``live_counts``, ``live_ids``, and ``mass_snapshots``
    describe the post-apply state.  The record has no method that can emit an
    operation or feed a policy component.
    """

    event_index: int
    applied_ops: Mapping[str, tuple[Any, ...]]
    live_counts: Mapping[str, int]
    mass_snapshots: Mapping[str, Tensor]
    prune_ages: Mapping[str, Tensor]
    rent_thresholds: Mapping[str, float | None]
    live_ids: Mapping[str, Tensor] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.event_index, bool) or not isinstance(self.event_index, int):
            raise TypeError("event_index must be an int")
        if self.event_index < 0:
            raise ValueError("event_index must be non-negative")
        sites = set(self.live_counts)
        for values in (
            self.applied_ops,
            self.mass_snapshots,
            self.prune_ages,
            self.rent_thresholds,
        ):
            if set(values) != sites:
                raise ValueError("every AuditRecord site mapping must have equal keys")
        ids = self.live_ids
        if ids is None:
            ids = {
                site: torch.arange(int(self.live_counts[site]), dtype=torch.int64)
                for site in sites
            }
        if set(ids) != sites:
            raise ValueError("live_ids must have the same site keys as live_counts")

        copied_ops: dict[str, tuple[Any, ...]] = {}
        copied_counts: dict[str, int] = {}
        copied_mass: dict[str, Tensor] = {}
        copied_ages: dict[str, Tensor] = {}
        copied_thresholds: dict[str, float | None] = {}
        copied_ids: dict[str, Tensor] = {}
        for site in sorted(sites):
            if not isinstance(site, str) or not site:
                raise ValueError("audit sites must be non-empty strings")
            count = self.live_counts[site]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("live counts must be non-negative ints")
            mass = _tensor_copy(self.mass_snapshots[site]).reshape(-1)
            site_ids = _tensor_copy(ids[site]).reshape(-1)
            ages = _tensor_copy(self.prune_ages[site]).reshape(-1)
            if site_ids.dtype != torch.int64 or ages.dtype != torch.int64:
                raise TypeError("audit IDs and ages must have dtype int64")
            if mass.numel() != count or site_ids.numel() != count:
                raise ValueError("live IDs and mass must align with live count")
            threshold = self.rent_thresholds[site]
            if threshold is not None:
                threshold = float(threshold)
                if not isfinite(threshold) or threshold < 0:
                    raise ValueError("rent thresholds must be finite and non-negative")
            copied_ops[site] = tuple(deepcopy(op) for op in self.applied_ops[site])
            copied_counts[site] = count
            copied_mass[site] = mass
            copied_ages[site] = ages
            copied_thresholds[site] = threshold
            copied_ids[site] = site_ids
        object.__setattr__(self, "applied_ops", copied_ops)
        object.__setattr__(self, "live_counts", copied_counts)
        object.__setattr__(self, "mass_snapshots", copied_mass)
        object.__setattr__(self, "prune_ages", copied_ages)
        object.__setattr__(self, "rent_thresholds", copied_thresholds)
        object.__setattr__(self, "live_ids", copied_ids)

    # Short aliases keep interactive analysis readable without changing the
    # explicit serialized field names above.
    @property
    def ops(self) -> Mapping[str, tuple[Any, ...]]:
        return self.applied_ops

    @property
    def k_live(self) -> Mapping[str, int]:
        return self.live_counts

    @property
    def mass(self) -> Mapping[str, Tensor]:
        return self.mass_snapshots


@runtime_checkable
class AuditSubscriber(Protocol):
    """A one-way sink. Engine code may only push aggregate event records."""

    def push(self, record: AuditRecord) -> None: ...


@dataclass(frozen=True)
class RentMargin:
    """Distribution summary of mass divided by the event's rent threshold."""

    event_index: int
    site: str
    minimum: float | None
    median: float | None


@dataclass(frozen=True)
class LossRecord:
    update_step: int
    value: float


class EconomyAudit:
    """Read-only aggregate economy of a lifecycle run.

    Thrash follows ``framework_conn_5c_report.md``'s lineage-age convention: a
    prune is young when its age is below ``maturity_events``.  The v4 audit
    denominator is all prunes.  The default maturity boundary is the
    *earliest legal* rent prune age for a :class:`~cstf.policy.courts.RentCourt`
    with ``strikes`` consecutive rent failures after ``immunity_events``:
    ``immunity_events + strikes - 1``.  Below that age no rent-court prune can
    legally occur at all (see ``immune_prunes``), and at exactly that age the
    prune is the court working as designed, not thrash.  Callers without a
    strikes-bearing court (e.g. ``MagnitudeCourt``) should leave ``strikes``
    at its default of 1, which reduces the boundary to ``immunity_events``.

    ``record_loss`` accepts arbitrary user-supplied loss readings solely for
    retrospective realized-profit analysis.  Those readings are never passed
    to the engine, a schedule, a proposer, an allocator, or a court.
    """

    def __init__(
        self,
        *,
        immunity_events: int = 3,
        strikes: int = 1,
        maturity_events: int | None = None,
    ) -> None:
        for name, value in (
            ("immunity_events", immunity_events),
            ("strikes", strikes),
            (
                "maturity_events",
                immunity_events + strikes - 1
                if maturity_events is None
                else maturity_events,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.immunity_events = immunity_events
        self.strikes = strikes
        self.maturity_events = (
            immunity_events + strikes - 1
            if maturity_events is None
            else maturity_events
        )
        self._records: list[AuditRecord] = []
        self._losses: list[LossRecord] = []

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    @property
    def losses(self) -> tuple[LossRecord, ...]:
        return tuple(self._losses)

    def push(self, record: AuditRecord) -> None:
        if not isinstance(record, AuditRecord):
            raise TypeError("record must be an AuditRecord")
        if self._records and record.event_index <= self._records[-1].event_index:
            raise ValueError("audit event indices must be strictly increasing")
        self._records.append(record)

    def record_loss(self, update_step: int, value: float) -> None:
        if isinstance(update_step, bool) or not isinstance(update_step, int):
            raise TypeError("update_step must be an int")
        if update_step < 0:
            raise ValueError("update_step must be non-negative")
        reading = float(value)
        if not isfinite(reading):
            raise ValueError("loss value must be finite")
        self._losses.append(LossRecord(update_step, reading))

    @property
    def K(self) -> dict[str, tuple[int, ...]]:
        sites = sorted(
            {site for record in self._records for site in record.live_counts}
        )
        return {
            site: tuple(record.live_counts[site] for record in self._records)
            for site in sites
        }

    @property
    def k(self) -> dict[str, tuple[int, ...]]:
        return self.K

    def k_series(self, site: str) -> tuple[int, ...]:
        try:
            return self.K[site]
        except KeyError as exc:
            raise KeyError(f"unknown audited site {site!r}") from exc

    @staticmethod
    def _op_size(op: Any) -> int:
        if isinstance(op, SynapseBirth):
            return int(op.w.numel())
        if isinstance(op, (SynapseDeath, NeuronUngate, NeuronRetire)):
            return int(op.ids.numel())
        return 0

    @property
    def churn_by_site(self) -> dict[str, tuple[int, ...]]:
        sites = sorted(
            {site for record in self._records for site in record.applied_ops}
        )
        return {
            site: tuple(
                sum(
                    self._op_size(op)
                    for op in record.applied_ops[site]
                    if isinstance(
                        op,
                        (SynapseBirth, SynapseDeath, NeuronUngate, NeuronRetire),
                    )
                )
                for record in self._records
            )
            for site in sites
        }

    @property
    def churn(self) -> tuple[int, ...]:
        by_site = self.churn_by_site
        return tuple(
            sum(values[index] for values in by_site.values())
            for index in range(len(self._records))
        )

    @property
    def prune_count(self) -> int:
        return sum(
            int(ages.numel())
            for record in self._records
            for ages in record.prune_ages.values()
        )

    @property
    def thrash_count(self) -> int:
        return sum(
            int((ages < self.maturity_events).sum())
            for record in self._records
            for ages in record.prune_ages.values()
        )

    @property
    def thrash_rate(self) -> float:
        total = self.prune_count
        return self.thrash_count / total if total else 0.0

    @property
    def immune_prunes(self) -> int:
        """Count prunes younger than ``immunity_events``; must always be 0.

        The engine's runtime immunity check rejects any court decision that
        would prune an entity below ``immunity_events``, so this is a
        read-only invariant witness, not a tunable metric.
        """
        return sum(
            int((ages < self.immunity_events).sum())
            for record in self._records
            for ages in record.prune_ages.values()
        )

    @property
    def thrash_rate_by_site(self) -> dict[str, float]:
        sites = sorted(
            {site for record in self._records for site in record.prune_ages}
        )
        result: dict[str, float] = {}
        for site in sites:
            columns = [record.prune_ages[site] for record in self._records]
            total = sum(int(column.numel()) for column in columns)
            young = sum(
                int((column < self.maturity_events).sum()) for column in columns
            )
            result[site] = young / total if total else 0.0
        return result

    @property
    def last_structural_event(self) -> int | None:
        structural = [
            record.event_index
            for record in self._records
            if any(record.applied_ops[site] for site in record.applied_ops)
        ]
        return structural[-1] if structural else None

    @property
    def quiescence_event(self) -> int | None:
        return self.last_structural_event

    @property
    def quiescence_step(self) -> int | None:
        """Compatibility name; structural time here is the event index."""
        return self.last_structural_event

    @property
    def rent_margins(self) -> dict[str, tuple[RentMargin, ...]]:
        sites = sorted(
            {site for record in self._records for site in record.mass_snapshots}
        )
        result: dict[str, tuple[RentMargin, ...]] = {}
        for site in sites:
            summaries: list[RentMargin] = []
            for record in self._records:
                mass = record.mass_snapshots[site].to(dtype=torch.float64)
                threshold = record.rent_thresholds[site]
                if mass.numel() == 0 or threshold is None:
                    minimum = median = None
                elif threshold == 0:
                    minimum = median = float("inf")
                else:
                    ratio = mass / threshold
                    minimum = float(ratio.min())
                    median = float(torch.quantile(ratio, 0.5))
                summaries.append(RentMargin(record.event_index, site, minimum, median))
            result[site] = tuple(summaries)
        return result

    @property
    def rent_margin(self) -> dict[str, tuple[RentMargin, ...]]:
        return self.rent_margins
