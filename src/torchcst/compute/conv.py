"""torchcst.compute.conv — relative CST conv (署名のみ確定)。"""

from __future__ import annotations

from torch import Tensor, nn

from ..contracts import Kernel
from ..storage.neuron import NeuronStore
from ..storage.synapse import SynapseStore


class CSTConv2d(nn.Module):
    """relative CST: 受容野内相対座標に原子を置き空間位置で共有。
    channel 側は k 共有低ランク (channel 混合は CST 化しない — cst 研究
    repo の境界地図に従う)。実装は未着手・署名のみ確定。"""

    def __init__(self, in_neurons: NeuronStore, out_neurons: NeuronStore,
                 synapses: SynapseStore, kernel: Kernel,
                 rf_size: int, channel_rank: int): ...

    def forward(self, x: Tensor) -> Tensor: ...
