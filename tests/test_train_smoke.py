"""1〜2 層 CSTLinear MLP の toy 回帰訓練スモークテスト。

途中で SynapseDeath → SynapseBirth を適用しても訓練が継続して壊れない
(nan にならず loss が下がり続ける) ことを暗黙に検証する。param object 自体は
不変 (行の書き換えのみ) なので Adam の state と param の対応は崩れない設計に
なっているはず — それをこのテストで実地確認する。
"""

from __future__ import annotations

import torch
from torch import nn

from torchcst.compute.kernels import GaussianKernel
from torchcst.compute.linear import CSTLinear
from torchcst.storage.neuron import NeuronStore
from torchcst.storage.synapse import SynapseBirth, SynapseDeath, SynapseStore


def _collect_params(objs):
    """store.parameters() / kernel.global_params() を id ベースで重複排除
    しつつ集める (同じ NeuronStore/kernel を複数層で共有する場合に備える)。"""
    seen: set[int] = set()
    params: list[nn.Parameter] = []
    for obj in objs:
        if hasattr(obj, "parameters"):
            ps = obj.parameters()
        elif hasattr(obj, "global_params"):
            ps = obj.global_params()
        else:
            ps = []
        for p in ps:
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _birth_random(store: SynapseStore, k: int, d_in: int, d_out: int,
                   w_init="randn"):
    s = torch.rand(k, d_in)
    t = torch.rand(k, d_out)
    w = torch.zeros(k) if w_init == "zeros" else 0.1 * torch.randn(k)
    store.apply([SynapseBirth(store.site, s=s, t=t, w=w)])


def _death_and_birth_smallest_mass(store: SynapseStore, n: int, d_in: int, d_out: int):
    view = store.view()
    order = torch.argsort(view.w.detach().abs())
    dying_ids = view.ids[order[:n]]
    store.apply([SynapseDeath(store.site, ids=dying_ids)])
    _birth_random(store, n, d_in, d_out, w_init="zeros")


def test_train_smoke_mlp_regression_survives_death_birth():
    torch.manual_seed(0)

    n_in, n_hidden, n_out = 16, 32, 1
    K = 256

    in_neurons = NeuronStore("in", torch.linspace(0, 1, n_in).unsqueeze(-1))
    hidden_neurons = NeuronStore(
        "hidden", torch.linspace(0, 1, n_hidden).unsqueeze(-1), gated=True
    )
    out_neurons = NeuronStore("out", torch.linspace(0, 1, n_out).unsqueeze(-1))

    syn1 = SynapseStore("l1", d_in=1, d_out=1, capacity=K)
    syn2 = SynapseStore("l2", d_in=1, d_out=1, capacity=K)
    _birth_random(syn1, K, 1, 1)
    _birth_random(syn2, K, 1, 1)

    kernel1 = GaussianKernel(0.15, learnable=True)
    kernel2 = GaussianKernel(0.15, learnable=True)

    layer1 = CSTLinear(in_neurons, hidden_neurons, syn1, kernel1)
    layer2 = CSTLinear(hidden_neurons, out_neurons, syn2, kernel2)
    model = nn.Sequential(layer1, nn.GELU(), layer2)

    params = _collect_params(
        [in_neurons, hidden_neurons, out_neurons, syn1, syn2, kernel1, kernel2]
    )
    opt = torch.optim.Adam(params, lr=2e-2)

    def make_batch(batch_size=64):
        x = torch.randn(batch_size, n_in)
        y = torch.sin(x.sum(dim=-1, keepdim=True))
        return x, y

    eval_x, eval_y = make_batch(256)

    with torch.no_grad():
        initial_loss = ((model(eval_x) - eval_y) ** 2).mean().item()

    n_steps = 300
    mutate_at = 150
    losses = []
    for step in range(n_steps):
        x, y = make_batch(64)
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        assert not torch.isnan(loss), f"loss went nan at step {step}"

        if step == mutate_at:
            _death_and_birth_smallest_mass(syn1, 10, 1, 1)
            # capacity == k_live (満杯) の syn2 でも同様に death→birth できる
            _death_and_birth_smallest_mass(syn2, 10, 1, 1)

    with torch.no_grad():
        final_loss = ((model(eval_x) - eval_y) ** 2).mean().item()

    assert not any(torch.isnan(torch.tensor(losses)))
    assert final_loss < initial_loss / 3, (
        f"expected significant improvement: initial={initial_loss}, final={final_loss}"
    )
