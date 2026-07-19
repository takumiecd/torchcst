"""Forward計算が利用するkernel契約。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol

from torch import Tensor, nn

if TYPE_CHECKING:
    from ..storage.synapse import SynapseStore


class Kernel(Protocol):
    def install(self, store: "SynapseStore") -> None: ...
    def global_params(self) -> Iterable[nn.Parameter]: ...
    def __call__(
        self, query: Tensor, centers: Tensor, extras: dict[str, Tensor]
    ) -> Tensor: ...
