"""CST forwardで使うkernel family。"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ..storage.synapse import SynapseStore


class GaussianKernel(nn.Module):
    def __init__(
        self,
        sigma: float,
        *,
        learnable: bool = True,
        per_atom: bool = False,
    ):
        super().__init__()
        if per_atom:
            raise NotImplementedError("per-atom sigma is not implemented")
        self.per_atom = per_atom
        log_sigma = torch.log(torch.as_tensor(float(sigma)))
        if learnable:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

    def install(self, store: SynapseStore) -> None:
        return None

    def global_params(self) -> Iterable[nn.Parameter]:
        if isinstance(self.log_sigma, nn.Parameter):
            return [self.log_sigma]
        return []

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        extras: dict[str, Tensor],
    ) -> Tensor:
        diff = query.unsqueeze(1) - centers.unsqueeze(0)
        squared_distance = diff.pow(2).sum(dim=-1)
        sigma = self.log_sigma.exp()
        return torch.exp(-squared_distance / (2 * sigma * sigma))


class TriangularKernel(nn.Module):
    def install(self, store: SynapseStore) -> None:
        return None

    def global_params(self) -> Iterable[nn.Parameter]:
        return []

    def forward(
        self,
        query: Tensor,
        centers: Tensor,
        extras: dict[str, Tensor],
    ) -> Tensor:
        raise NotImplementedError("TriangularKernel is not implemented")
