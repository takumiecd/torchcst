"""CSTLinear.forward が明示的に組んだ dense 行列と一致することを確認する
要のテスト: W_ij = Σ_k w_k κ(μ_i^in − s_k) κ(μ_j^out − t_k)。
"""

from __future__ import annotations

import torch

from torchcst.forward.kernels import GaussianKernel
from torchcst.forward.linear import CSTLinear
from torchcst.storage.neuron import NeuronStore
from torchcst.storage.synapse import SynapseBirth, SynapseStore


def _dense_weight(mu_in, mu_out, s, t, w, sigma):
    # W[j, i] = sum_k w_k * kappa(mu_in_i - s_k) * kappa(mu_out_j - t_k)
    diff_in = mu_in.unsqueeze(1) - s.unsqueeze(0)     # [N_in, K, d]
    k_in = torch.exp(-diff_in.pow(2).sum(-1) / (2 * sigma ** 2))   # [N_in, K]
    diff_out = mu_out.unsqueeze(1) - t.unsqueeze(0)   # [N_out, K, d]
    k_out = torch.exp(-diff_out.pow(2).sum(-1) / (2 * sigma ** 2))  # [N_out, K]
    W = torch.einsum("ik,jk,k->ji", k_in, k_out, w)   # [N_out, N_in]
    return W


def test_cstlinear_matches_dense_matrix_plain():
    torch.manual_seed(0)
    n_in, n_out, K, d, sigma = 8, 6, 20, 1, 0.3

    mu_in = torch.linspace(0, 1, n_in).unsqueeze(-1)
    mu_out = torch.linspace(0, 1, n_out).unsqueeze(-1)
    in_neurons = NeuronStore("in", mu_in)
    out_neurons = NeuronStore("out", mu_out)

    synapses = SynapseStore("l1", d_in=d, d_out=d, capacity=K)
    s = torch.rand(K, d)
    t = torch.rand(K, d)
    w = torch.randn(K)
    synapses.apply([SynapseBirth("l1", s=s, t=t, w=w)])

    kernel = GaussianKernel(sigma, learnable=False)
    layer = CSTLinear(in_neurons, out_neurons, synapses, kernel)

    x = torch.randn(4, n_in)
    y = layer(x)

    W = _dense_weight(mu_in, mu_out, s, t, w, sigma)
    y_expected = x @ W.t()

    assert torch.allclose(y, y_expected, atol=1e-5)


def test_cstlinear_matches_dense_matrix_gated():
    torch.manual_seed(1)
    n_in, n_out, K, d, sigma = 8, 6, 20, 1, 0.3

    mu_in = torch.linspace(0, 1, n_in).unsqueeze(-1)
    mu_out = torch.linspace(0, 1, n_out).unsqueeze(-1)
    in_neurons = NeuronStore("in", mu_in, gated=True)
    out_neurons = NeuronStore("out", mu_out, gated=True)

    synapses = SynapseStore("l1", d_in=d, d_out=d, capacity=K)
    s = torch.rand(K, d)
    t = torch.rand(K, d)
    w = torch.randn(K)
    synapses.apply([SynapseBirth("l1", s=s, t=t, w=w)])

    kernel = GaussianKernel(sigma, learnable=False)
    layer = CSTLinear(in_neurons, out_neurons, synapses, kernel)

    # gate は初期値 ones だが、非自明にするため手で書き換える
    with torch.no_grad():
        in_neurons.c.copy_(torch.rand(n_in) + 0.5)
        out_neurons.c.copy_(torch.rand(n_out) + 0.5)

    x = torch.randn(4, n_in)
    y = layer(x)

    W = _dense_weight(mu_in, mu_out, s, t, w, sigma)
    y_expected = (x * in_neurons.c) @ W.t()
    y_expected = y_expected * out_neurons.c

    assert torch.allclose(y, y_expected, atol=1e-5)


def test_cstlinear_separate_kernel_in_out():
    torch.manual_seed(2)
    n_in, n_out, K, d = 5, 4, 10, 1
    sigma_in, sigma_out = 0.2, 0.5

    mu_in = torch.linspace(0, 1, n_in).unsqueeze(-1)
    mu_out = torch.linspace(0, 1, n_out).unsqueeze(-1)
    in_neurons = NeuronStore("in", mu_in)
    out_neurons = NeuronStore("out", mu_out)

    synapses = SynapseStore("l1", d_in=d, d_out=d, capacity=K)
    s = torch.rand(K, d)
    t = torch.rand(K, d)
    w = torch.randn(K)
    synapses.apply([SynapseBirth("l1", s=s, t=t, w=w)])

    kernel_in = GaussianKernel(sigma_in, learnable=False)
    kernel_out = GaussianKernel(sigma_out, learnable=False)
    layer = CSTLinear(in_neurons, out_neurons, synapses, kernel_in, kernel_out)

    x = torch.randn(3, n_in)
    y = layer(x)

    diff_in = mu_in.unsqueeze(1) - s.unsqueeze(0)
    k_in = torch.exp(-diff_in.pow(2).sum(-1) / (2 * sigma_in ** 2))
    diff_out = mu_out.unsqueeze(1) - t.unsqueeze(0)
    k_out = torch.exp(-diff_out.pow(2).sum(-1) / (2 * sigma_out ** 2))
    W = torch.einsum("ik,jk,k->ji", k_in, k_out, w)
    y_expected = x @ W.t()

    assert torch.allclose(y, y_expected, atol=1e-5)
