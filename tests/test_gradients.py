"""backward で s, t, w, log_sigma, (gated の c) 全てに grad が流れること、
live でない slot の grad は 0 であることを確認する。"""

from __future__ import annotations

import torch

from torchcst.compute.kernels import GaussianKernel
from torchcst.compute.linear import CSTLinear
from torchcst.storage.neuron import NeuronStore
from torchcst.storage.synapse import SynapseBirth, SynapseStore


def test_gradients_flow_to_all_learnables_plain():
    torch.manual_seed(0)
    n_in, n_out, K_live, capacity, d = 6, 4, 8, 12, 1

    mu_in = torch.linspace(0, 1, n_in).unsqueeze(-1)
    mu_out = torch.linspace(0, 1, n_out).unsqueeze(-1)
    in_neurons = NeuronStore("in", mu_in)
    out_neurons = NeuronStore("out", mu_out)

    synapses = SynapseStore("l1", d_in=d, d_out=d, capacity=capacity)
    s = torch.rand(K_live, d)
    t = torch.rand(K_live, d)
    w = torch.randn(K_live)
    synapses.apply(SynapseBirth("l1", s=s, t=t, w=w))

    kernel = GaussianKernel(0.3, learnable=True)
    layer = CSTLinear(in_neurons, out_neurons, synapses, kernel)

    x = torch.randn(5, n_in)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    assert synapses.s.grad is not None
    assert synapses.t.grad is not None
    assert synapses.w.grad is not None
    assert kernel.log_sigma.grad is not None

    live_slots = synapses._slots.live_slots
    dead_mask = torch.ones(capacity, dtype=torch.bool)
    dead_mask[live_slots] = False

    assert dead_mask.any()  # capacity > k_live なので dead slot が存在する
    assert torch.all(synapses.s.grad[dead_mask] == 0)
    assert torch.all(synapses.t.grad[dead_mask] == 0)
    assert torch.all(synapses.w.grad[dead_mask] == 0)

    # live slot は (高確率で) 非ゼロ勾配を持つ
    assert synapses.w.grad[live_slots].abs().sum() > 0


def test_gradients_flow_to_gate_when_gated():
    torch.manual_seed(1)
    n_in, n_out, K, d = 5, 3, 6, 1

    mu_in = torch.linspace(0, 1, n_in).unsqueeze(-1)
    mu_out = torch.linspace(0, 1, n_out).unsqueeze(-1)
    in_neurons = NeuronStore("in", mu_in, gated=True)
    out_neurons = NeuronStore("out", mu_out, gated=True)

    synapses = SynapseStore("l1", d_in=d, d_out=d, capacity=K)
    synapses.apply(SynapseBirth("l1", s=torch.rand(K, d), t=torch.rand(K, d),
                                 w=torch.randn(K)))

    kernel = GaussianKernel(0.3, learnable=True)
    layer = CSTLinear(in_neurons, out_neurons, synapses, kernel)

    x = torch.randn(4, n_in)
    y = layer(x)
    y.pow(2).sum().backward()

    assert in_neurons.c.grad is not None
    assert out_neurons.c.grad is not None
    assert not torch.all(in_neurons.c.grad == 0)
    assert not torch.all(out_neurons.c.grad == 0)
