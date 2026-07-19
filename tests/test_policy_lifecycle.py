"""PyTorch-native Policy lifecycleの最小契約テスト。"""

from __future__ import annotations

import torch

import torchcst as tc


def _layer() -> tuple[tc.CSTLinear, tc.SynapseStore]:
    in_store = tc.NeuronStore(
        "in", torch.linspace(0, 1, 4).unsqueeze(-1)
    )
    out_store = tc.NeuronStore(
        "out", torch.linspace(0, 1, 3).unsqueeze(-1)
    )
    synapses = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=8)
    synapses.apply([
        tc.SynapseBirth(
            "l1",
            s=torch.rand(4, 1),
            t=torch.rand(4, 1),
            w=torch.randn(4),
        )
    ])
    return (
        tc.CSTLinear(
            in_store,
            out_store,
            synapses,
            tc.GaussianKernel(0.2),
        ),
        synapses,
    )


def _engine(policy: tc.Policy) -> tuple[tc.CSTEngine, tc.SynapseStore]:
    layer, synapses = _layer()
    engine = tc.CSTEngine(
        layer,
        lambda params: torch.optim.Adam(params, lr=1e-3),
        policy,
    )
    return engine, synapses


def test_set_prepares_without_gradient_capture_and_emits_site_batch():
    policy = tc.cSET(
        ["l1"],
        schedule=tc.PeriodicSchedule(every=1, until=2, fraction=lambda _: 0.25),
    )
    engine, synapses = _engine(policy)

    engine.backward(engine.model(torch.randn(2, 4)).sum())
    engine.optimizer.step()
    ops = engine.step()

    assert len(ops) == 2
    assert synapses.view().version == 2


def test_rigl_capture_is_policy_owned_and_produces_candidates():
    policy = tc.cRigL(
        ["l1"],
        schedule=tc.PeriodicSchedule(every=1, until=2, fraction=lambda _: 0.25),
        pool=4,
    )
    engine, synapses = _engine(policy)

    engine.backward(engine.model(torch.randn(2, 4)).sum())
    engine.optimizer.step()
    ops = engine.step()

    assert len(ops) == 2
    assert synapses.view().version == 2
    assert policy.candidates["l1"].snapshot().scores.shape == (4,)
