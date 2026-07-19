"""CSTLinear forwardと、そのPyTorch-native gradient capture。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..backward import CapturePoint, LinearGradRecord
from ..storage.base import EntityStore
from ..storage.neuron import NeuronStore
from ..storage.synapse import SynapseStore
from .base import Kernel


class LinearGradientProvider:
    """CSTLinearのkernel/gateを使って任意座標のdL/dwを評価する。"""

    def __init__(self, module: "CSTLinear"):
        self._module = module

    def weight_gradients(
        self,
        input: Tensor,
        grad_output: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            in_view = self._module.in_neurons.view()
            out_view = self._module.out_neurons.view()

            k_in = self._module.kernel_in(in_view.mu, source, {})
            k_out = self._module.kernel_out(out_view.mu, target, {})

            x = input * in_view.gate if in_view.gate is not None else input
            g = (
                grad_output * out_view.gate
                if out_view.gate is not None
                else grad_output
            )
            return ((x @ k_in) * (g @ k_out)).sum(dim=0)


class CSTLinear(nn.Module):
    """NeuronStore × SynapseStore × Kernelで連続疎線形層を構成する。"""

    def __init__(
        self,
        in_neurons: NeuronStore,
        out_neurons: NeuronStore,
        synapses: SynapseStore,
        kernel_in: Kernel,
        kernel_out: Kernel | None = None,
    ):
        super().__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.synapses = synapses
        self.kernel_in = kernel_in
        self.kernel_out = kernel_in if kernel_out is None else kernel_out

        self.kernel_in.install(self.synapses)
        if self.kernel_out is not self.kernel_in:
            self.kernel_out.install(self.synapses)

        self.grad_capture = CapturePoint(
            self.synapses.site, LinearGradRecord
        )
        self._gradient_provider = LinearGradientProvider(self)

    def forward(self, input: Tensor) -> Tensor:
        in_view = self.in_neurons.view()
        out_view = self.out_neurons.view()
        synapse_view = self.synapses.view()

        k_in = self.kernel_in(
            in_view.mu, synapse_view.s, synapse_view.extras
        )
        k_out = self.kernel_out(
            out_view.mu, synapse_view.t, synapse_view.extras
        )

        x = input * in_view.gate if in_view.gate is not None else input
        hidden = (x @ k_in) * synapse_view.w
        output = hidden @ k_out.t()
        if out_view.gate is not None:
            output = output * out_view.gate

        if self.grad_capture.active and output.requires_grad:
            site = self.synapses.site
            version = synapse_view.version
            input_detached = input.detach()
            capture = self.grad_capture
            provider = self._gradient_provider

            def emit(grad_output: Tensor) -> None:
                capture.emit(LinearGradRecord(
                    site=site,
                    version=version,
                    input=input_detached,
                    grad_output=grad_output.detach(),
                    gradients=provider,
                ))

            output.register_hook(emit)

        return output

    def stores(self) -> tuple[EntityStore, ...]:
        return self.in_neurons, self.out_neurons, self.synapses

    def capture_points(self) -> tuple[CapturePoint, ...]:
        return (self.grad_capture,)
