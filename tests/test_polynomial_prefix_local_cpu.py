"""Independent autograd checks for the isolated polynomial contraction screen."""

import numpy as np
import pytest
import torch

from benchmarks.research.polynomial_prefix_local_cpu import evaluate


@pytest.mark.parametrize("method", ["direct", "grouped_direct", "prefix"])
@pytest.mark.parametrize("sigma_value", [0.1, 0.4])
def test_local_polynomial_contraction_against_torch(method, sigma_value):
    x = (np.arange(160) + 0.5) * 0.1
    mu = np.array([6.4, 6.437, 7.1, 9.57])
    sigma = np.full(len(mu), sigma_value)
    values = np.random.default_rng(47).normal(size=(2, len(x)))
    got, _ = evaluate(x, values, mu, sigma, 4, method)
    for j in range(len(mu)):
        tm = torch.tensor(mu[j], dtype=torch.float64, requires_grad=True)
        ts = torch.tensor(sigma[j], dtype=torch.float64, requires_grad=True)
        q = (1 - ((torch.tensor(x) - tm) / (np.sqrt(11) * ts)) ** 2).clamp_min(0) ** 4
        q = q / q.norm().clamp_min(1e-12)
        for b in range(2):
            h = q @ torch.tensor(values[b])
            dm, ds = torch.autograd.grad(h, (tm, ts), retain_graph=True)
            np.testing.assert_allclose(
                got[j, :, b], [h.item(), dm.item(), ds.item()], atol=2e-9, rtol=2e-9
            )
