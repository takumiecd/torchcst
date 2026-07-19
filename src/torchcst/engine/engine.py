"""Entity-neutralなPolicy lifecycleとmutation routing。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from ..backward import BackwardContext, CapturePoint
from ..policy.base import Policy
from ..policy.binding import PolicyBinding
from ..policy.schedule import Clock
from ..storage.base import EntityStore, Op


class _OptimizerStateFollower:
    """再利用slotに残るoptimizer stateをmutation時に消去する。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: Iterable[nn.Parameter],
    ):
        self._optimizer = optimizer
        self._params = tuple(params)

    def _zero_rows(self, slots: torch.Tensor) -> None:
        for parameter in self._params:
            state = self._optimizer.state.get(parameter)
            if not state:
                continue
            for value in state.values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == parameter.shape[0]
                ):
                    value.index_fill_(0, slots.to(value.device), 0.0)

    def grow(self, new_capacity: int) -> None:
        raise NotImplementedError("optimizer state growth is not implemented")

    def on_birth(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_death(self, slots: torch.Tensor) -> None:
        self._zero_rows(slots)

    def on_merge(
        self,
        src_slots: torch.Tensor,
        dst_slots: torch.Tensor,
        mass: torch.Tensor,
    ) -> None:
        raise NotImplementedError("optimizer state merge is not implemented")

    def on_remap(self, old_to_new: torch.Tensor) -> None:
        raise NotImplementedError("optimizer state remap is not implemented")


class CSTEngine:
    """Model resourcesをbindし、PolicyのMutationPlanをStoreへrouteする。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: (
            torch.optim.Optimizer
            | Callable[[list[nn.Parameter]], torch.optim.Optimizer]
        ),
        policy: Policy,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        if world_size != 1:
            raise NotImplementedError("distributed execution is not implemented")

        self.model = model
        self.policy = policy
        self.rank = rank
        self.world_size = world_size
        self._step = 0
        self._op_log: list[tuple[int, Op]] = []
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        self._stores = self._collect_stores(model)
        captures = self._collect_captures(model)
        parameters = self._collect_parameters(model, self._stores.values())
        self.optimizer = self._make_optimizer(optimizer, parameters)

        for store in self._stores.values():
            params = tuple(store.parameters())
            if params:
                store.followers().subscribe(
                    _OptimizerStateFollower(self.optimizer, params)
                )

        policy.prepare(PolicyBinding(self._stores, captures))

    @staticmethod
    def _collect_stores(model: nn.Module) -> dict[str, EntityStore]:
        stores: dict[str, EntityStore] = {}
        for module in model.modules():
            provider = getattr(module, "stores", None)
            if provider is None:
                continue
            for store in provider():
                if not isinstance(store, EntityStore):
                    raise TypeError(
                        f"{type(module).__name__}.stores() returned "
                        f"{type(store).__name__}, expected EntityStore"
                    )
                current = stores.get(store.site)
                if current is not None and current is not store:
                    raise ValueError(
                        f"site {store.site!r} is bound to different stores"
                    )
                stores[store.site] = store
        return stores

    @staticmethod
    def _collect_captures(model: nn.Module) -> tuple[CapturePoint, ...]:
        captures: list[CapturePoint] = []
        seen: set[int] = set()
        for module in model.modules():
            provider = getattr(module, "capture_points", None)
            if provider is None:
                continue
            for capture in provider():
                if not isinstance(capture, CapturePoint):
                    raise TypeError(
                        f"{type(module).__name__}.capture_points() returned "
                        f"{type(capture).__name__}, expected CapturePoint"
                    )
                if id(capture) not in seen:
                    seen.add(id(capture))
                    captures.append(capture)
        return tuple(captures)

    @staticmethod
    def _collect_parameters(
        model: nn.Module,
        stores: Iterable[EntityStore],
    ) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in model.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        for store in stores:
            for parameter in store.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    @staticmethod
    def _make_optimizer(
        optimizer: (
            torch.optim.Optimizer
            | Callable[[list[nn.Parameter]], torch.optim.Optimizer]
        ),
        parameters: list[nn.Parameter],
    ) -> torch.optim.Optimizer:
        if isinstance(optimizer, torch.optim.Optimizer):
            existing = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            missing = [p for p in parameters if id(p) not in existing]
            if missing:
                optimizer.add_param_group({"params": missing})
            return optimizer
        if callable(optimizer):
            return optimizer(parameters)
        raise TypeError("optimizer must be an Optimizer or parameter factory")

    def step(self) -> list[Op]:
        """optimizer.step()後にPolicyを評価し、MutationPlanを適用する。"""
        request = self.policy.schedule.poll(Clock(self._step), self._rng)
        if request is None:
            self._step += 1
            return []

        plan = self.policy.step(request)
        sites = [batch.site for batch in plan.batches]
        if len(sites) != len(set(sites)):
            raise ValueError("MutationPlan must contain at most one batch per site")

        applied: list[Op] = []
        for batch in plan.batches:
            try:
                store = self._stores[batch.site]
            except KeyError as exc:
                raise ValueError(
                    f"MutationPlan targets unknown site {batch.site!r}"
                ) from exc
            store.apply(batch.ops)
            applied.extend(batch.ops)
            self._op_log.extend((store.version, op) for op in batch.ops)

        self._step += 1
        return applied

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        """Run autograd and dispatch captured records after backward completes."""
        with BackwardContext(self._collect_captures(self.model)):
            loss.backward(*args, **kwargs)

    def close(self) -> None:
        self.policy.close()

    def op_log(self) -> list[tuple[int, Op]]:
        return list(self._op_log)

    def save(self, path: str) -> None:
        raise NotImplementedError("CSTEngine.save is not implemented")

    def load(self, path: str) -> None:
        raise NotImplementedError("CSTEngine.load is not implemented")
