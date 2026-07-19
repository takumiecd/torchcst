"""Storageが公開するentity-neutral契約。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from torch import Tensor, nn


@dataclass(frozen=True)
class View:
    """Storeの論理状態を表すread-only viewの共通部分。"""

    site: str
    version: int
    capacity: int


class Op(Protocol):
    """Store固有mutation命令に共通するrouting情報。"""

    site: str


class Follower(Protocol):
    """slot-indexedな付随状態をStore mutationへ追従させる契約。"""

    def grow(self, new_capacity: int) -> None: ...
    def on_birth(self, slots: Tensor) -> None: ...
    def on_death(self, slots: Tensor) -> None: ...
    def on_merge(
        self, src_slots: Tensor, dst_slots: Tensor, mass: Tensor
    ) -> None: ...
    def on_remap(self, old_to_new: Tensor) -> None: ...


class EntityStore(ABC):
    """EngineがStoreについて利用できる最小契約。"""

    site: str

    @property
    @abstractmethod
    def version(self) -> int: ...

    @abstractmethod
    def live_ids(self) -> Tensor: ...

    @abstractmethod
    def view(self) -> View: ...

    @abstractmethod
    def apply(self, ops: Sequence[Op]) -> None: ...

    @abstractmethod
    def parameters(self) -> Iterable[nn.Parameter]: ...

    @abstractmethod
    def followers(self): ...

    @abstractmethod
    def canonical_state(self) -> dict: ...

    @abstractmethod
    def load_state(self, state: dict) -> None: ...
