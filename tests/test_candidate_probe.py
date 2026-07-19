"""CandidateProbeの新しいGradRecord APIを検証する。"""

from __future__ import annotations

import torch

import torchcst as tc


def test_candidate_probe_observes_linear_grad_record_and_resamples_on_mutation():
    torch.manual_seed(3)
    ins = tc.NeuronStore("in", torch.rand(8, 1))
    outs = tc.NeuronStore("out", torch.rand(6, 1))
    syn = tc.SynapseStore("l1", 1, 1, 20)
    syn.apply([tc.SynapseBirth("l1", torch.rand(20, 1), torch.rand(20, 1), torch.randn(20))])
    layer = tc.CSTLinear(ins, outs, syn, tc.GaussianKernel(0.3))
    probe = tc.policies.CandidateProbe(pool=16, decay=0.0, seed=7)
    layer.grad_capture.subscribe(lambda record: probe.observe(record, syn.view()))

    layer(torch.randn(4, 8)).pow(2).sum().backward()
    before = probe.snapshot()
    assert before.scores.shape == (16,)

    old_s = before.s.clone()
    syn.apply([tc.SynapseDeath("l1", ids=syn.view().ids[:2])])
    layer.zero_grad(set_to_none=True)
    layer(torch.randn(4, 8)).pow(2).sum().backward()

    after = probe.snapshot()
    assert not torch.equal(after.s, old_s)
    assert after.scores.shape == (16,)
