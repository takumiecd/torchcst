"""Shared tile prototype and finite-grid norm against direct Torch values."""

import pytest
import torch

from benchmarks.research.hierarchical_fourier_cpu import (
    contract_tiles,
    make_layout,
    sampled_norm_squared,
)
from torchcst import FourierPeakSpec, HierarchicalFourierPeriodSpec
from torchcst._backends.torch.kernels.hierarchical_fourier import values

DT = torch.float64


def settings(modes):
    return HierarchicalFourierPeriodSpec(
        minimum_tile_width=0.05 if modes == 4 else 0.1,
        minimum_half_period=0.0125 if modes == 4 else 0.025,
        relative_width_floor=0.7 if modes == 4 else 0.54,
    )


@pytest.mark.parametrize("modes", [4, 8])
@pytest.mark.parametrize("support", ["centered", "whole_tiles"])
def test_dirichlet_norm_and_all_parameter_gradients(modes, support):
    shape, spec = FourierPeakSpec(modes=modes), settings(modes)
    sites = torch.arange(256, dtype=DT) * 0.1
    for step in range(3):
        mu = torch.tensor(
            [0.137, 1.371, 7.319, 12.137, 100.0], dtype=DT, requires_grad=True
        )
        I = torch.tensor([0.1, 0.2, 0.81, 10.0, 0.1], dtype=DT) + step * 0.003
        I.requires_grad_()
        actual = sampled_norm_squared(
            shape, spec, mu, I, sites=len(sites), support=support
        )
        expected = values(shape, spec, sites, mu, I, support=support).square().sum(0)
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=2e-10)
        cot = torch.tensor([0.7, -0.1, 0.3, 0.9, 0.2], dtype=DT)
        ga = torch.autograd.grad(actual @ cot, (mu, I), retain_graph=True)
        ge = torch.autograd.grad(expected @ cot, (mu, I))
        for a, b in zip(ga, ge):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("modes", [4, 8])
def test_actual_bottom_up_shared_h_dx_center_width_and_l2(modes):
    profile, spec = FourierPeakSpec(modes=modes), settings(modes)
    generator = torch.Generator().manual_seed(115)
    x = torch.arange(256, dtype=DT) * 0.1
    for step in range(3):
        mu = (
            torch.tensor([0.137, 1.371, 7.319, 12.137], dtype=DT, requires_grad=True)
            + step * 0.007
        )
        I = (
            torch.tensor([0.1, 0.2, 0.81, 10.0], dtype=DT, requires_grad=True)
            + step * 0.003
        )
        inp = torch.randn(2, 256, dtype=DT, generator=generator, requires_grad=True)
        cot = torch.randn(2, 4, dtype=DT, generator=generator)
        layout = make_layout(profile, spec, mu, I, sites=len(x))
        H, payload = contract_tiles(profile, spec, inp, mu, I, layout)
        gram = sampled_norm_squared(profile, spec, mu, I, sites=len(x))
        H = H / gram.clamp_min(1e-16).sqrt()
        raw = values(profile, spec, x, mu, I, support="whole_tiles")
        target = inp @ (raw / raw.norm(dim=0).clamp_min(1e-8))
        torch.testing.assert_close(H, target, atol=4e-11, rtol=2e-10)
        actual = torch.autograd.grad((H * cot).sum(), (inp, mu, I), retain_graph=True)
        expected = torch.autograd.grad((target * cot).sum(), (inp, mu, I))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=3e-9, rtol=2e-8)
        assert payload["dag_moment_reals"] == 2 * layout["stats"]["dag_moment_reals"]
        assert payload["leaf_basis_reals"] == layout["stats"]["leaf_basis_reals"]
