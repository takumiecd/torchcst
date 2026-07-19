"""GradEMAがPyTorch-native LinearGradRecordと一致することを確認する。"""

from __future__ import annotations

import torch

import torchcst as tc


def _check(gated: bool) -> None:
    torch.manual_seed(1 if gated else 0)
    n_in, n_out, k = 8, 6, 20
    ins = tc.NeuronStore("in", torch.rand(n_in, 1), gated=gated)
    outs = tc.NeuronStore("out", torch.rand(n_out, 1), gated=gated)
    syn = tc.SynapseStore("l1", 1, 1, k)
    syn.apply([tc.SynapseBirth("l1", torch.rand(k, 1), torch.rand(k, 1), torch.randn(k))])
    if gated:
        with torch.no_grad():
            ins.c.copy_(torch.rand(n_in) + 0.5)
            outs.c.copy_(torch.rand(n_out) + 0.5)

    layer = tc.CSTLinear(ins, outs, syn, tc.GaussianKernel(0.3))
    ema = tc.policies.GradEMA(decay=0.0)
    layer.grad_capture.subscribe(lambda record: ema.observe(record, syn.view()))

    layer(torch.randn(5, n_in)).pow(2).sum().backward()
    scores = ema.snapshot()
    view = syn.view()
    expected = syn.w.grad.index_select(0, syn._slots.live_slots).abs()
    assert torch.equal(scores.ids, view.ids)
    assert torch.allclose(scores.scores, expected, atol=1e-5)


def test_gradema_matches_autograd_plain():
    _check(False)


def test_gradema_matches_autograd_gated():
    _check(True)
