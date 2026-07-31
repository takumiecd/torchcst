"""Cross-store two-phase apply: tickets, the store contract, prepare/commit-all.

A structural event may touch several stores at once (e.g. a neuron ungate plus
its incident synapse births). Atomicity across stores comes from the calling
order, not from locking: every store must :meth:`~EntityStore.prepare`
completely -- validating and freezing its batch without mutation -- before any
store :meth:`~EntityStore.commit`\\ s. :func:`prepare_all` and
:func:`commit_all` are the two halves of that discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence


@dataclass(eq=False)
class Ticket:
    """Single-use commit voucher bound to one store at one structural version.

    ``batch`` holds the store-private frozen mutation values; its concrete
    type differs per store and is opaque to everyone else.
    """

    store: EntityStore
    version: int
    batch: Any
    empty: bool = False
    _used: bool = False


class EntityStore(Protocol):
    """The minimal shared surface synapse and neuron stores expose for two-phase apply."""

    site: str

    def prepare(self, ops: Sequence[Any]) -> Ticket: ...

    def commit(self, ticket: Ticket) -> None: ...

    def _validate_ticket(self, ticket: Ticket) -> None: ...


def validate_ticket(ticket: Ticket, store: EntityStore, version: int) -> None:
    """Shared head of every store's ticket validation.

    Each store's ``_validate_ticket`` calls this, then adds its own
    batch-specific tail checks.
    """
    if not isinstance(ticket, Ticket):
        raise TypeError("ticket must be a Ticket")
    if ticket.store is not store:
        raise ValueError("ticket belongs to another store")
    if ticket._used:
        raise RuntimeError("ticket has already been committed")
    if ticket.version != version:
        raise RuntimeError("ticket is stale")


def prepare_all(
    plans: Iterable[tuple[EntityStore, Sequence[Any]]],
) -> tuple[Ticket, ...]:
    """Prepare every store's batch; tickets exist only if all stores succeed."""
    return tuple(store.prepare(ops) for store, ops in plans)


def commit_all(tickets: Iterable[Ticket]) -> None:
    """Re-validate every ticket, then commit store by store.

    The pre-pass rejects a stale or duplicated ticket before any store has
    mutated, so a bad joint event aborts whole.
    """
    tickets = tuple(tickets)
    stores = [ticket.store for ticket in tickets]
    if len({id(store) for store in stores}) != len(stores):
        raise ValueError("commit_all accepts at most one ticket per store")
    for store, ticket in zip(stores, tickets):
        store._validate_ticket(ticket)
    for store, ticket in zip(stores, tickets):
        store.commit(ticket)
