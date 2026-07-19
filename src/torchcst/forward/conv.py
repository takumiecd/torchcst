"""relative CST convolution (interface only)。"""

from __future__ import annotations

from torch import Tensor, nn

from ..storage.neuron import NeuronStore
from ..storage.synapse import SynapseStore
from .base import Kernel


class CSTConv2d(nn.Module):
    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel: Kernel,
        rf_size: int,
        channel_rank: int,
    ):
        super().__init__()
        raise NotImplementedError("CSTConv2d is not implemented")

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("CSTConv2d is not implemented")
