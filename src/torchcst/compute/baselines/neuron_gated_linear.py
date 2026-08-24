"""Optional neuron gating for entry and rank-one control families."""

from __future__ import annotations

from torch import Tensor, nn

from torchcst.storage import NeuronStore, SynapseStore

from ..capture import BackwardContext
from .entry_linear import EntryLinear
from .rank_one_linear import RankOneLinear


class NeuronGatedLinear(nn.Module):
    """Compose neuron gates around a neuron-independent control linear.

    `EntryLinear` and `RankOneLinear` intentionally know only their synapse
    stores. This wrapper is the opt-in topology boundary for experiments that
    need dormant/live/retired neuron gates or endpoint-aware response policies.
    `CSTLinear` does not use this wrapper because neuron coordinates are an
    intrinsic part of its factor representation.
    """

    def __init__(
        self,
        linear: EntryLinear | RankOneLinear,
        in_neurons: NeuronStore | None = None,
        out_neurons: NeuronStore | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(linear, (EntryLinear, RankOneLinear)):
            raise TypeError("linear must be an EntryLinear or RankOneLinear")
        self._validate_neurons(in_neurons, linear.in_features, "in_neurons")
        self._validate_neurons(out_neurons, linear.out_features, "out_neurons")
        if in_neurons is None and out_neurons is None:
            raise ValueError("at least one neuron endpoint is required")

        self.linear = linear
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.capture_site = linear.capture_site

    @property
    def store(self) -> SynapseStore:
        """Expose the wrapped store without registering it a second time."""
        return self.linear.store

    @staticmethod
    def _validate_neurons(store: NeuronStore | None, width: int, name: str) -> None:
        if store is not None and not isinstance(store, NeuronStore):
            raise TypeError(f"{name} must be a NeuronStore or None")
        if store is not None and store.n_max != width:
            raise ValueError(f"{name}.n_max must equal its feature width")

    @staticmethod
    def _gate(store: NeuronStore | None, reference: Tensor) -> Tensor | None:
        if store is None:
            return None
        return store.gate_vector().to(device=reference.device, dtype=reference.dtype)

    def set_backward_context(self, context: BackwardContext | None) -> None:
        self.linear.set_backward_context(context)

    @property
    def capture_enabled(self) -> bool:
        return self.linear.capture_enabled

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        in_gate = self._gate(self.in_neurons, x)
        output = self.linear(x if in_gate is None else x * in_gate)
        out_gate = self._gate(self.out_neurons, output)
        return output if out_gate is None else output * out_gate

    def atom_grads(self, x: Tensor, g_out: Tensor) -> Tensor:
        """Delegate captured, already-gated facts to the wrapped linear."""
        return self.linear.atom_grads(x, g_out)

    def dense_weight(self) -> Tensor:
        """Materialize the gated dense map solely for tests and debugging."""
        result = self.linear.dense_weight()
        if self.in_neurons is not None:
            result = result * self.in_neurons.gate_vector().to(result)
        if self.out_neurons is not None:
            result = result * self.out_neurons.gate_vector().to(result)[:, None]
        return result
