"""torchcst.compute.linear — 横断合成: synapse と neuron が出会う場所。"""

from __future__ import annotations

from torch import Tensor, nn

from ..contracts import EntityStore, Kernel
from ..storage.neuron import NeuronStore
from ..storage.synapse import SynapseStore


class CSTLinear(nn.Module):
    """in/out の NeuronStore × SynapseStore × Kernel の合成。

    forward は View のみから:
      y = gate_out ⊙ (((x ⊙ gate_in) @ K_in) * w) @ K_out.T
      K_in = κ_in(μ_in ⊖ s) [N_in,K], K_out = κ_out(μ_out ⊖ t) [N_out,K]
    (N_in×N_out 非実体化)。View は store.version が動いた時のみ再取得。

    backward hook は Observation を publish する「だけ」— この module は
    Op を発行できない (三角形の辺制約)。hook 内は per-item 同期禁止・
    バッチ一括ベクトル演算のみ。
    """

    def __init__(self, in_neurons: NeuronStore, out_neurons: NeuronStore,
                 synapses: SynapseStore, kernel_in: Kernel,
                 kernel_out: Kernel | None = None):
        """kernel_out=None なら kernel_in を両側共有。in/out に同じ
        NeuronStore を渡せば座標共有、別 store なら分離。"""
        ...

    def forward(self, x: Tensor) -> Tensor: ...
    def stores(self) -> list[EntityStore]: ...
