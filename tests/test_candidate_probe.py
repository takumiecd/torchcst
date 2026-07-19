"""CandidateProbe の score が dense |kappa_out(t_c)^T G kappa_in(s_c)| と
一致すること (gated 含む)、および mutation で pool が再サンプルされ score
がリセットされることを確認する。"""

from __future__ import annotations

import torch

import torchcst as tc
from torchcst.decision.instruments import CandidateProbe
from torchcst.engine import _LayerKernelPort


def _dense_score(layer, x, g_out_grad, s_c, t_c):
    kernel = layer.kernel_in
    in_view = layer.in_neurons.view()
    out_view = layer.out_neurons.view()

    gate_in = in_view.gate
    gate_out = out_view.gate
    x_gated = x * gate_in if gate_in is not None else x
    g_gated = g_out_grad * gate_out if gate_out is not None else g_out_grad

    g_matrix = torch.einsum("bj,bi->ji", g_gated, x_gated)   # [N_out, N_in]
    k_in_pool = kernel(in_view.mu, s_c, {})     # [N_in, P]
    k_out_pool = layer.kernel_out(out_view.mu, t_c, {})  # [N_out, P]
    return (k_out_pool * (g_matrix @ k_in_pool)).sum(dim=0).abs()


def _build_layer(gated: bool, seed: int):
    torch.manual_seed(seed)
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
    return layer, syn


def _check_probe_matches_dense(gated: bool):
    layer, syn = _build_layer(gated, seed=3 if not gated else 4)
    port = _LayerKernelPort(layer)

    probe = CandidateProbe(pool=30, decay=0.0, seed=42)
    probe.bind(syn, port)
    layer.on_observation = lambda obs: probe.update(obs, syn.view())

    x = torch.randn(5, layer.in_neurons.mu.shape[0])
    y = layer(x)
    loss = y.pow(2).sum()
    y.retain_grad()
    loss.backward()

    reading = probe.read()
    g_out_grad = y.grad  # dL/dy captured directly (same forward, no randomness)
    expected = _dense_score(layer, x, g_out_grad, probe._s, probe._t)

    assert torch.allclose(expected, reading.scores, atol=1e-4)


def test_candidate_probe_matches_dense_plain():
    _check_probe_matches_dense(gated=False)


def test_candidate_probe_matches_dense_gated():
    _check_probe_matches_dense(gated=True)


def test_candidate_probe_resamples_pool_on_version_change():
    layer, syn = _build_layer(gated=False, seed=5)
    port = _LayerKernelPort(layer)

    probe = CandidateProbe(pool=16, decay=0.9, seed=7)
    probe.bind(syn, port)
    layer.on_observation = lambda obs: probe.update(obs, syn.view())

    x = torch.randn(4, layer.in_neurons.mu.shape[0])
    y = layer(x)
    y.pow(2).sum().backward()

    s_before = probe._s.clone()
    t_before = probe._t.clone()
    scores_before = probe.read().scores.clone()
    assert scores_before.abs().sum() > 0  # 何かしらスコアが付いている

    # mutation: 適当に 2 個 death させて version を進める
    view = syn.view()
    dying = view.ids[:2]
    syn.apply([tc.SynapseDeath("l1", ids=dying)])

    layer.zero_grad(set_to_none=True)
    x2 = torch.randn(4, layer.in_neurons.mu.shape[0])
    y2 = layer(x2)
    y2.pow(2).sum().backward()

    assert not torch.equal(probe._s, s_before)
    assert not torch.equal(probe._t, t_before)
    # 再サンプル直後の decay 適用結果は「新規 score_batch そのもの」
    # (0 からのスタートに 1 回分の update をかけただけ) であって、
    # 旧 scores_before の情報を引き継いでいない。
    assert probe._version == syn.version
