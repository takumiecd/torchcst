"""torchcst.compute.kernels — κ の族。

global パラメタは自前 nn.Parameter。per-atom パラメタが必要なら
install(store) で SynapseStore.add_extra を呼んで列を確保する。
"""

from __future__ import annotations

from torch import Tensor, nn

from ..storage.synapse import SynapseStore


class GaussianKernel(nn.Module):
    """σ を global スカラー or per-atom 列で持てる Gaussian。"""

    def __init__(self, sigma: float, *, learnable: bool = True,
                 per_atom: bool = False): ...

    def install(self, store: SynapseStore) -> None: ...
    def forward(self, query: Tensor, centers: Tensor,
                extras: dict[str, Tensor]) -> Tensor: ...


class TriangularKernel(nn.Module):
    """引数最小・コンパクトサポートの例。"""

    def install(self, store: SynapseStore) -> None: ...
    def forward(self, query: Tensor, centers: Tensor,
                extras: dict[str, Tensor]) -> Tensor: ...
