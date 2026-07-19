"""torchcst.compute.kernels — κ の族。

global パラメタは自前 nn.Parameter。per-atom パラメタが必要なら
install(store) で SynapseStore.add_extra を呼んで列を確保する。
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ..storage.synapse import SynapseStore


class GaussianKernel(nn.Module):
    """σ を global スカラー or per-atom 列で持てる Gaussian。

    v0 スコープ: global σ のみ (per_atom=True は NotImplementedError)。
    σ の正値保証のため log_sigma を学習対象にする。
    """

    def __init__(self, sigma: float, *, learnable: bool = True,
                 per_atom: bool = False):
        super().__init__()
        if per_atom:
            raise NotImplementedError("v0: per_atom sigma not implemented")
        self.per_atom = per_atom
        log_sigma = torch.log(torch.as_tensor(float(sigma)))
        if learnable:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

    def install(self, store: SynapseStore) -> None:
        # global σ のみなので per-atom 列は不要 (no-op)。
        return None

    def global_params(self) -> Iterable[nn.Parameter]:
        if isinstance(self.log_sigma, nn.Parameter):
            return [self.log_sigma]
        return []

    def forward(self, query: Tensor, centers: Tensor,
                extras: dict[str, Tensor]) -> Tensor:
        """κ(query_i − center_k) = exp(-||query_i − center_k||² / (2σ²))。
        query [N, d], centers [K, d] → [N, K]。"""
        diff = query.unsqueeze(1) - centers.unsqueeze(0)   # [N, K, d]
        sq_dist = diff.pow(2).sum(dim=-1)                   # [N, K]
        sigma = self.log_sigma.exp()
        return torch.exp(-sq_dist / (2 * sigma * sigma))


class TriangularKernel(nn.Module):
    """引数最小・コンパクトサポートの例。"""

    def install(self, store: SynapseStore) -> None: ...
    def forward(self, query: Tensor, centers: Tensor,
                extras: dict[str, Tensor]) -> Tensor: ...
