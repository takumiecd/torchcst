"""三角形が初めて一周する要のテスト: CSTEngine + cSET で toy 回帰を実際に
回し、mutation を挟んでも訓練が壊れないこと・op_log の整合・Adam state の
ゼロ化を確認する。"""

from __future__ import annotations

import torch
from torch import nn

import torchcst as tc
from torchcst.storage.synapse import SynapseBirth, SynapseDeath


def _new_ids(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    before_set = set(before.tolist())
    new = [i for i in after.tolist() if i not in before_set]
    return torch.tensor(new, dtype=torch.int64)


def test_policy_schedule_returns_named_stage_without_hidden_phase_state():
    policy = tc.policies.cSET(sites=["l1"], dt=10, t_end=20)

    assert policy.schedule(1) == []
    stages = policy.schedule(10)

    assert len(stages) == 1
    assert isinstance(stages[0], tc.DecisionStage)
    assert stages[0].name == "rewire"
    assert stages[0].sites == ("l1",)
    assert not hasattr(policy, "_step")
    assert not hasattr(policy, "_pending_n")


def test_engine_e2e_cset_toy_regression():
    torch.manual_seed(0)

    n_in, n_hidden, n_out = 16, 32, 1
    K = 256

    in_neurons = tc.NeuronStore("in", torch.linspace(0, 1, n_in).unsqueeze(-1))
    hidden_neurons = tc.NeuronStore(
        "hidden", torch.linspace(0, 1, n_hidden).unsqueeze(-1), gated=True
    )
    out_neurons = tc.NeuronStore("out", torch.linspace(0, 1, n_out).unsqueeze(-1))

    syn1 = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=K)
    syn2 = tc.SynapseStore("l2", d_in=1, d_out=1, capacity=K)
    syn1.apply([SynapseBirth(
        "l1", s=torch.rand(K, 1), t=torch.rand(K, 1), w=0.1 * torch.randn(K)
    )])
    syn2.apply([SynapseBirth(
        "l2", s=torch.rand(K, 1), t=torch.rand(K, 1), w=0.1 * torch.randn(K)
    )])

    kernel1 = tc.GaussianKernel(0.15)
    kernel2 = tc.GaussianKernel(0.15)
    layer1 = tc.CSTLinear(in_neurons, hidden_neurons, syn1, kernel1)
    layer2 = tc.CSTLinear(hidden_neurons, out_neurons, syn2, kernel2)
    model = nn.Sequential(layer1, nn.GELU(), layer2)

    dt, t_end, n_steps = 100, 350, 400
    policy = tc.policies.cSET(sites=["l1", "l2"], dt=dt, t_end=t_end,
                               frac=lambda t: 0.05)

    opt_factory = lambda params: torch.optim.Adam(params, lr=1e-2)
    engine = tc.CSTEngine(model, opt_factory, policy, seed=0)

    def make_batch(bs=64):
        x = torch.randn(bs, n_in)
        y = torch.sin(x.sum(dim=-1, keepdim=True))
        return x, y

    eval_x, eval_y = make_batch(256)
    with torch.no_grad():
        initial_loss = ((model(eval_x) - eval_y) ** 2).mean().item()

    expected_trigger_steps = [s for s in range(n_steps) if s % dt == 0 and s < t_end]
    assert expected_trigger_steps == [0, 100, 200, 300]

    for step in range(n_steps):
        x, y = make_batch(64)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        engine.optimizer.zero_grad()
        loss.backward()
        engine.optimizer.step()

        assert not torch.isnan(loss)

        ids_before = {"l1": syn1.view().ids, "l2": syn2.view().ids}
        k_before = {"l1": syn1._slots.k_live, "l2": syn2._slots.k_live}

        ops = engine.step()

        if ops:
            # assert 5: engine.step() の戻り値と op_log の末尾が整合する
            log_tail = engine.op_log()[-len(ops):]
            assert [op for (_v, op) in log_tail] == ops
            # 同一 site の death/birth は一つの store transaction なので、
            # op log 上も同じ structure version を共有する。
            for site in ("l1", "l2"):
                site_versions = [v for v, op in log_tail if op.site == site]
                assert len(set(site_versions)) == 1

            for site, syn in (("l1", syn1), ("l2", syn2)):
                # assert 3: death n + birth n で k_live が保存される
                assert syn._slots.k_live == k_before[site]

                # assert 4: 新生原子の slot は Adam state が厳密に 0
                ids_after = syn.view().ids
                new_ids = _new_ids(ids_before[site], ids_after)
                assert new_ids.numel() > 0
                new_slots = syn._slots.slots_of(new_ids)

                for p in (syn.s, syn.t, syn.w):
                    st = engine.optimizer.state.get(p)
                    assert st is not None
                    assert torch.all(st["exp_avg"][new_slots] == 0)
                    assert torch.all(st["exp_avg_sq"][new_slots] == 0)

    with torch.no_grad():
        final_loss = ((model(eval_x) - eval_y) ** 2).mean().item()

    # assert 1: loss が有意に下がる・nan なし
    assert final_loss < initial_loss / 3

    # assert 2: death/birth の回数が schedule から計算できる期待値と一致する
    n_sites = 2
    deaths = [op for _v, op in engine.op_log() if isinstance(op, SynapseDeath)]
    births = [op for _v, op in engine.op_log() if isinstance(op, SynapseBirth)]
    assert len(deaths) == len(expected_trigger_steps) * n_sites
    assert len(births) == len(expected_trigger_steps) * n_sites


def test_engine_rejects_conflicting_site_stores():
    n = 4
    neurons_a = tc.NeuronStore("shared_in", torch.linspace(0, 1, n).unsqueeze(-1))
    neurons_b = tc.NeuronStore("shared_in", torch.linspace(0, 1, n).unsqueeze(-1))
    out_a = tc.NeuronStore("out_a", torch.linspace(0, 1, 2).unsqueeze(-1))
    out_b = tc.NeuronStore("out_b", torch.linspace(0, 1, 2).unsqueeze(-1))

    syn_a = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=4)
    syn_b = tc.SynapseStore("l2", d_in=1, d_out=1, capacity=4)
    syn_a.apply([SynapseBirth(
        "l1", s=torch.rand(4, 1), t=torch.rand(4, 1), w=torch.zeros(4)
    )])
    syn_b.apply([SynapseBirth(
        "l2", s=torch.rand(4, 1), t=torch.rand(4, 1), w=torch.zeros(4)
    )])

    layer_a = tc.CSTLinear(neurons_a, out_a, syn_a, tc.GaussianKernel(0.2))
    layer_b = tc.CSTLinear(neurons_b, out_b, syn_b, tc.GaussianKernel(0.2))

    class TwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = layer_a
            self.b = layer_b

        def forward(self, x):
            return self.a(x) + self.b(x)

    model = TwoBranch()
    policy = tc.policies.cSET(sites=["l1", "l2"])

    import pytest
    with pytest.raises(ValueError):
        tc.CSTEngine(model, lambda params: torch.optim.Adam(params, lr=1e-3), policy)


def test_engine_accepts_shared_neuron_store_same_object():
    n_in, n_hidden, n_out = 4, 6, 2
    in_neurons = tc.NeuronStore("in", torch.linspace(0, 1, n_in).unsqueeze(-1))
    hidden_neurons = tc.NeuronStore("hidden", torch.linspace(0, 1, n_hidden).unsqueeze(-1), gated=True)
    out_neurons = tc.NeuronStore("out", torch.linspace(0, 1, n_out).unsqueeze(-1))

    syn1 = tc.SynapseStore("l1", d_in=1, d_out=1, capacity=6)
    syn2 = tc.SynapseStore("l2", d_in=1, d_out=1, capacity=6)
    syn1.apply([SynapseBirth(
        "l1", s=torch.rand(6, 1), t=torch.rand(6, 1), w=torch.zeros(6)
    )])
    syn2.apply([SynapseBirth(
        "l2", s=torch.rand(6, 1), t=torch.rand(6, 1), w=torch.zeros(6)
    )])

    layer1 = tc.CSTLinear(in_neurons, hidden_neurons, syn1, tc.GaussianKernel(0.2))
    layer2 = tc.CSTLinear(hidden_neurons, out_neurons, syn2, tc.GaussianKernel(0.2))
    model = nn.Sequential(layer1, nn.GELU(), layer2)

    policy = tc.policies.cSET(sites=["l1", "l2"])
    engine = tc.CSTEngine(model, lambda params: torch.optim.Adam(params, lr=1e-3), policy)
    assert engine._stores["hidden"] is hidden_neurons
