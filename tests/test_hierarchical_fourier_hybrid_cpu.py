"""Algorithm routing only: same values and all gradients through thresholds."""

import pytest
import torch

from benchmarks.research.hierarchical_fourier_hybrid_cpu import hybrid_contract
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec
from torchcst._backends.torch.kernels.hierarchical_fourier import values


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("cutoff", [0, 2, 8, 32, 1000])
def test_dynamic_direct_shared_same_profile_values_and_all_gradients(modes, cutoff):
    spec = HierarchicalFourierPeriodSpec(
        minimum_tile_width=0.05 if modes == 4 else 0.1,
        minimum_half_period=0.0125 if modes == 4 else 0.025,
        relative_width_floor=0.7 if modes == 4 else 0.54,
    )
    profile = FourierPeakSpec(modes=modes)
    x = torch.arange(256, dtype=torch.float64) * 0.1
    g = torch.Generator().manual_seed(37)
    for step in range(3):
        I = (
            (torch.tensor([0.1, 0.332, 1.0, 10.0], dtype=torch.float64) + step * 0.003)
            .clamp(0.1, 10)
            .requires_grad_()
        )
        mu = (
            torch.tensor([0.137, 1.371, 7.319, 12.137], dtype=torch.float64)
            + step * 0.007
        ).requires_grad_()
        inp = torch.randn(2, 256, dtype=torch.float64, generator=g, requires_grad=True)
        cot = torch.randn(2, 4, dtype=torch.float64, generator=g)
        h, n, _ = hybrid_contract(profile, spec, inp, mu, I, cutoff=cutoff)
        h = h / n.clamp_min(1e-16).sqrt()
        f = values(profile, spec, x, mu, I, support="whole_tiles")
        ref = inp @ (f / f.norm(dim=0).clamp_min(1e-8))
        torch.testing.assert_close(h, ref, atol=3e-11, rtol=3e-10)
        a = torch.autograd.grad((h * cot).sum(), (inp, mu, I), retain_graph=True)
        b = torch.autograd.grad((ref * cot).sum(), (inp, mu, I))
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, atol=3e-9, rtol=2e-8)
