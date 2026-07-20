"""Opt-in realized-profit adjudication and complete trial rollback."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import torch

from torchcst.storage import SynapseBirth, SynapseDeath, SynapseMerge

from .contract import InstrumentSpec


class TrialTransaction:
    """Deep snapshot for the isolated, opt-in profit-trial path.

    The transaction covers all store parameters/buffers and structural extra
    state, optimizer state, stateful courts and instruments, the retired
    registry, engine clock/op-log, the engine generator, and process RNG state.
    It is intentionally heavyweight.  The step-8 MVP performs zero polish
    updates inside a trial; future finite-polish implementations can use this
    same boundary without weakening rollback.
    """

    def __init__(self, engine: Any) -> None:
        if not hasattr(engine, "stores") or not hasattr(engine, "policy"):
            raise TypeError("engine must provide structural engine state")
        self.engine = engine
        self._active = True
        self._store_states = {
            site: deepcopy(store.state_dict())
            for site, store in engine.stores.items()
        }
        self._optimizer_state = (
            None
            if engine.optimizer is None
            else deepcopy(engine.optimizer.state_dict())
        )
        self._component_states: list[tuple[Any, Any]] = []
        components = (
            engine.policy.retention,
            engine.policy.neuron_retention,
            engine.policy.profit,
        )
        seen: set[int] = set()
        for component in components:
            exporter = getattr(component, "state_dict", None)
            if component is None or exporter is None or id(component) in seen:
                continue
            seen.add(id(component))
            self._component_states.append((component, deepcopy(exporter())))
        self._instrument_states: list[tuple[Any, Any]] = []
        for site_instruments in engine.instruments.values():
            for instrument in dict.fromkeys(site_instruments.values()):
                exporter = getattr(instrument, "state_dict", None)
                if exporter is not None:
                    self._instrument_states.append(
                        (instrument, deepcopy(exporter()))
                    )
        self._registry_state = deepcopy(engine.registry.state_dict())
        self._clock = deepcopy(engine.clock)
        self._op_log = deepcopy(engine._op_log)
        self._rng_state = engine.rng.get_state().clone()
        self._torch_rng_state = torch.random.get_rng_state().clone()
        self._cuda_rng_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )

    @property
    def active(self) -> bool:
        return self._active

    def commit(self) -> None:
        """Keep trial state and discard rollback authority."""
        if not self._active:
            raise RuntimeError("trial transaction is already closed")
        self._active = False

    def rollback(self) -> None:
        """Restore every captured owner exactly once."""
        if not self._active:
            raise RuntimeError("trial transaction is already closed")
        for site, state in self._store_states.items():
            self.engine.stores[site].load_state_dict(deepcopy(state))
        if self._optimizer_state is not None:
            assert self.engine.optimizer is not None
            self.engine.optimizer.load_state_dict(deepcopy(self._optimizer_state))
        for component, state in self._component_states:
            component.load_state_dict(deepcopy(state))
        for instrument, state in self._instrument_states:
            instrument.load_state_dict(deepcopy(state))
        self.engine.registry.load_state_dict(deepcopy(self._registry_state))
        self.engine.clock = deepcopy(self._clock)
        self.engine._op_log[:] = deepcopy(self._op_log)
        self.engine.rng.set_state(self._rng_state.clone())
        torch.random.set_rng_state(self._torch_rng_state.clone())
        if self._cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(
                [state.clone() for state in self._cuda_rng_states]
            )
        self._active = False

    def __enter__(self) -> TrialTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._active:
            self.rollback()


class TrialSession:
    """Objective capability sealed inside one transaction.

    ``begin`` evaluates the pre-trial objective once. ``evaluate_after`` is
    single-use.  Step 8 has no polish callback: the state between those reads
    differs only by the prepared/committed structural trial.
    """

    def __init__(
        self,
        objective: Callable[[], float],
        transaction: TrialTransaction,
    ) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable")
        if not isinstance(transaction, TrialTransaction):
            raise TypeError("transaction must be a TrialTransaction")
        self._objective = objective
        self.transaction = transaction
        self._loss_before: float | None = None
        self._loss_after: float | None = None

    @staticmethod
    def _reading(value: object) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("objective must return a scalar")
            value = value.detach().item()
        reading = float(value)  # type: ignore[arg-type]
        if not isfinite(reading):
            raise ValueError("objective must return a finite value")
        return reading

    def begin(self) -> float:
        if self._loss_before is not None:
            raise RuntimeError("trial objective-before was already evaluated")
        self._loss_before = self._reading(self._objective())
        return self._loss_before

    @property
    def loss_before(self) -> float:
        if self._loss_before is None:
            raise RuntimeError("trial session has not begun")
        return self._loss_before

    def evaluate_after(self) -> float:
        if self._loss_before is None:
            raise RuntimeError("trial session has not begun")
        if self._loss_after is not None:
            raise RuntimeError("trial objective-after was already evaluated")
        self._loss_after = self._reading(self._objective())
        return self._loss_after


@dataclass
class ProfitCourt:
    """Accept only positive realized profit, implementing CST rule S-5.

    ``profit = loss_before - loss_after - price`` and acceptance is strict:
    ``profit > min_profit``.  Therefore accepting only positive realized
    profit makes the price-inclusive objective non-increasing (theoretical
    rule S-5).  This court never triggers work; it adjudicates operations that
    a clock schedule and proposer have already selected.

    Prices use ``cost_rate * delta_params``.  A merge removes one atom per
    pair, so its price is negative ``cost_rate * atom_cost`` and acts as a
    resource saving.  ``cost_rate=0`` gives pure loss-delta adjudication.
    """

    min_profit: float = 0.0
    cost_rate: float = 0.0
    requires: tuple[InstrumentSpec, ...] = field(
        default=(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.min_profit = float(self.min_profit)
        self.cost_rate = float(self.cost_rate)
        if not isfinite(self.min_profit):
            raise ValueError("min_profit must be finite")
        if not isfinite(self.cost_rate) or self.cost_rate < 0:
            raise ValueError("cost_rate must be finite and non-negative")

    def price_for(
        self,
        trial_ops: Sequence[object],
        stores: Mapping[str, object],
    ) -> float:
        delta_params = 0
        for op in tuple(trial_ops):
            try:
                store = stores[op.site]  # type: ignore[attr-defined]
                atom_cost = int(store.spec.atom_cost)  # type: ignore[attr-defined]
            except (AttributeError, KeyError) as exc:
                raise ValueError("profit-priced op targets no synapse store") from exc
            if isinstance(op, SynapseMerge):
                delta_params -= int(op.id_pairs.shape[0]) * atom_cost
            elif isinstance(op, SynapseBirth):
                delta_params += int(op.w.numel()) * atom_cost
            elif isinstance(op, SynapseDeath):
                delta_params -= int(op.ids.numel()) * atom_cost
            else:
                raise TypeError(f"unsupported profit-priced op {type(op)!r}")
        return self.cost_rate * delta_params

    def adjudicate(
        self,
        trial_ops: Sequence[object],
        price: float,
        session: TrialSession,
    ) -> bool:
        """Evaluate once after apply, then commit or completely roll back."""
        tuple(trial_ops)  # Materialize once; selected operations are not ranked here.
        if not isinstance(session, TrialSession):
            raise TypeError("session must be a TrialSession")
        price = float(price)
        if not isfinite(price):
            raise ValueError("price must be finite")
        try:
            loss_after = session.evaluate_after()
            profit = session.loss_before - loss_after - price
            accepted = profit > self.min_profit
            if accepted:
                session.transaction.commit()
            else:
                session.transaction.rollback()
            return accepted
        except BaseException:
            if session.transaction.active:
                session.transaction.rollback()
            raise

    def state_dict(self) -> dict[str, str]:
        """ProfitCourt is intentionally stateless across trials."""
        return {"schema": "torchcst-profit-court-v1"}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "torchcst-profit-court-v1":
            raise ValueError("unsupported ProfitCourt state schema")

