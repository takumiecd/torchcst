"""No 2-site Fourier cache: byte-admission and exact mixed contraction."""

import pytest
import torch

from benchmarks.research.hierarchical_fourier_capacity_cpu import (
    capacity_contract,
    capacity_plan,
)
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec
from torchcst._backends.torch.kernels.hierarchical_fourier import values


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("minimum_sites", [32, 64, 128])
def test_large_cache_admission_preserves_all_gradients(modes, minimum_sites):
    spec = HierarchicalFourierPeriodSpec(
        minimum_tile_width=0.05 if modes == 4 else 0.2,
        minimum_half_period=0.0125 if modes == 4 else 0.05,
        relative_width_floor=0.7 if modes == 4 else 0.3,
    )
    profile = FourierPeakSpec(modes=modes)
    generator = torch.Generator().manual_seed(71)
    x = torch.arange(1024, dtype=torch.float64) * 0.1
    for step in range(2):
        I = (
            torch.tensor([0.1, 0.332, 8.0, 8.1, 8.2, 8.3], dtype=torch.float64)
            + step * 0.003
        ).requires_grad_()
        mu = (
            torch.tensor(
                [0.137, 1.371, 51.231, 51.243, 51.259, 51.271], dtype=torch.float64
            )
            + step * 0.007
        ).requires_grad_()
        inp = torch.randn(
            2, 1024, generator=generator, dtype=torch.float64, requires_grad=True
        )
        cot = torch.randn(2, 6, generator=generator, dtype=torch.float64)
        plan = capacity_plan(
            profile,
            spec,
            mu,
            I,
            width_cutoff=1.0,
            minimum_sites=minimum_sites,
            batch=2,
            storage_ratio=0.75,
            minimum_reuse=2,
        )
        h, n, stats = capacity_contract(profile, spec, inp, mu, I, plan=plan)
        y = h / n.clamp_min(1e-16).sqrt()
        f = values(profile, spec, x, mu, I, support="whole_tiles")
        target = inp @ (f / f.norm(dim=0).clamp_min(1e-8))
        torch.testing.assert_close(y, target, atol=4e-11, rtol=3e-10)
        a = torch.autograd.grad((y * cot).sum(), (inp, mu, I), retain_graph=True)
        b = torch.autograd.grad((target * cot).sum(), (inp, mu, I))
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, atol=3e-9, rtol=2e-8)
        if stats["stored_tiles"]:
            assert stats["min_stored_site_count"] >= minimum_sites
        if modes == 8:
            assert stats["stored_tiles"] > 0
