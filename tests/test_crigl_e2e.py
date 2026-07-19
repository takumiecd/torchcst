"""cRigL (MassEMA death + CandidateProbe birth) の完動テスト。cSET の e2e と
同構成で 400 step 訓練し、loss 収束・k_live 保存・op_log 整合に加えて
「birth された座標がその step の probe topk_coords と厳密一致する」ことを
確認する (統計的な近さでなく決定性そのものを検証する)。"""

from __future__ import annotations

import torch
from torch import nn

import torchcst as tc
from torchcst.storage.synapse import SynapseBirth, SynapseDeath


def test_crigl_e2e_toy_regression_and_birth_coord_determinism():
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
    syn1.apply(SynapseBirth("l1", s=torch.rand(K, 1), t=torch.rand(K, 1), w=0.1 * torch.randn(K)))
    syn2.apply(SynapseBirth("l2", s=torch.rand(K, 1), t=torch.rand(K, 1), w=0.1 * torch.randn(K)))

    kernel1 = tc.GaussianKernel(0.15)
    kernel2 = tc.GaussianKernel(0.15)
    layer1 = tc.CSTLinear(in_neurons, hidden_neurons, syn1, kernel1)
    layer2 = tc.CSTLinear(hidden_neurons, out_neurons, syn2, kernel2)
    model = nn.Sequential(layer1, nn.GELU(), layer2)

    dt, t_end, n_steps = 100, 350, 400
    policy = tc.policies.cRigL(sites=["l1", "l2"], dt=dt, t_end=t_end, pool=512,
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

        k_before = {"l1": syn1._slots.k_live, "l2": syn2._slots.k_live}

        ops = engine.step()

        if ops:
            log_tail = engine.op_log()[-len(ops):]
            assert [op for (_v, op) in log_tail] == ops

            for site, syn in (("l1", syn1), ("l2", syn2)):
                # k_live は death n + birth n で保存される
                assert syn._slots.k_live == k_before[site]

            for op in ops:
                if not isinstance(op, SynapseBirth):
                    continue
                # 決定性チェック: birth された座標は、その time-step で
                # probe が持っていた reading の topk_coords(n) と厳密一致する。
                # engine.step() 完了後〜次の backward までは probe の内部
                # 状態 (pool/scores) は誰にも更新されないので、ここで
                # 読み直しても decide() が見たのと同じスナップショットに
                # なる (byte-exact な再現性の検証であって、統計的な近さの
                # flaky なテストにはしていない)。
                probe = engine._instruments[f"probe:{op.site}"]
                n = op.s.shape[0]
                expected_s, expected_t = probe.read().topk_coords(n)
                assert torch.equal(op.s, expected_s)
                assert torch.equal(op.t, expected_t)

    with torch.no_grad():
        final_loss = ((model(eval_x) - eval_y) ** 2).mean().item()

    assert final_loss < initial_loss / 3

    n_sites = 2
    deaths = [op for _v, op in engine.op_log() if isinstance(op, SynapseDeath)]
    births = [op for _v, op in engine.op_log() if isinstance(op, SynapseBirth)]
    assert len(deaths) == len(expected_trigger_steps) * n_sites
    assert len(births) == len(expected_trigger_steps) * n_sites
