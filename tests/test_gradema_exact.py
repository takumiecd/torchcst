"""GradEMA が KernelPort 経由で計算する |dL/dw| が、autograd の
w.grad と厳密に一致することを確認する要のテスト (gate 込みの数式の証明)。
"""

from __future__ import annotations

import torch

import torchcst as tc
from torchcst.decision.instruments import GradEMA
from torchcst.engine import _LayerKernelPort


def _check_grad_ema_matches_autograd(gated: bool):
    torch.manual_seed(0 if not gated else 1)
    n_in, n_out, K, d = 8, 6, 20, 1

    in_neurons = tc.NeuronStore("in", torch.rand(n_in, d), gated=gated)
    out_neurons = tc.NeuronStore("out", torch.rand(n_out, d), gated=gated)
    syn = tc.SynapseStore("l1", d_in=d, d_out=d, capacity=K)
    syn.apply([tc.SynapseBirth("l1", s=torch.rand(K, d), t=torch.rand(K, d),
                                w=torch.randn(K))])

    if gated:
        with torch.no_grad():
            in_neurons.c.copy_(torch.rand(n_in) + 0.5)
            out_neurons.c.copy_(torch.rand(n_out) + 0.5)

    kernel = tc.GaussianKernel(0.3, learnable=True)
    layer = tc.CSTLinear(in_neurons, out_neurons, syn, kernel)
    port = _LayerKernelPort(layer)

    # decay=0.0 => 初回 update 後の ema はそのまま |grad| (平滑化なしで比較)
    inst = GradEMA(decay=0.0)
    inst.bind(syn, port)
    layer.on_observation = lambda obs: inst.update(obs, syn.view())

    x = torch.randn(5, n_in)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    view = syn.view()
    live_slots = syn._slots.live_slots
    expected = syn.w.grad.index_select(0, live_slots).abs()

    reading = inst.read()
    got = reading.value(view.ids)

    assert torch.allclose(expected, got, atol=1e-5)


def test_gradema_matches_autograd_plain():
    _check_grad_ema_matches_autograd(gated=False)


def test_gradema_matches_autograd_gated():
    _check_grad_ema_matches_autograd(gated=True)


def test_gradema_bind_without_port_raises():
    syn = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=4)
    syn.apply([tc.SynapseBirth("l1", s=torch.rand(4, 1), t=torch.rand(4, 1),
                                w=torch.zeros(4))])
    inst = GradEMA()
    try:
        inst.bind(syn, None)
        assert False, "expected ValueError when port is None"
    except ValueError:
        pass
