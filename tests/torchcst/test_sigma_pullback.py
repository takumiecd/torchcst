"""The sigma block: metric correctness, calibration, cap, and guards."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchcst import CSTPullbackAdam
from torchcst.compute import CSTLinear
from torchcst.optim.sigma import _bandwidth_jacobian_sq
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _site(*, atoms=1, normalized=True, seed=31, shared_factor=True,
          factor=None):
    store = SynapseStore(
        "sigma-test",
        1,
        1,
        atoms,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    store.apply([
        SynapseBirth(
            store.site,
            torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
            torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
            0.5 + torch.rand(atoms, generator=generator,
                             dtype=torch.float64),
            torch.arange(atoms, dtype=torch.int64),
        )
    ])
    inputs = NeuronStore(
        "sigma-input",
        7,
        mu=torch.linspace(-0.5, 1.5, 7, dtype=torch.float64).reshape(-1, 1),
        initial_live=7,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "sigma-output",
        6,
        mu=torch.linspace(-0.4, 1.4, 6, dtype=torch.float64).reshape(-1, 1),
        initial_live=6,
        dtype=torch.float64,
    )
    factor = GaussianFactor(0.35).double() if factor is None else factor
    module = CSTLinear(
        inputs,
        outputs,
        store,
        factor,
        factor_out=None if shared_factor else GaussianFactor(0.35).double(),
        gauge=L2NormalizedColumns() if normalized else None,
    )
    return module, store


@pytest.mark.parametrize("normalized", [True, False])
def test_single_atom_metric_matches_finite_differences(normalized):
    """With one atom the per-atom metric is exact: compare ||dW/dsigma||^2."""
    module, _store = _site(atoms=1, normalized=normalized,
                           shared_factor=False)
    sigma = module.factor_in.sigma
    h = 1e-5
    with torch.no_grad():
        sigma.add_(h)
        plus = module.dense_weight().clone()
        sigma.sub_(2 * h)
        minus = module.dense_weight().clone()
        sigma.add_(h)
    numeric = ((plus - minus) / (2 * h)).square().sum()
    analytic = _bandwidth_jacobian_sq(module, "in", sigma)
    torch.testing.assert_close(analytic, numeric, rtol=1e-4, atol=1e-10)


def test_first_sigma_step_is_calibrated_and_capped():
    module, _store = _site(atoms=3)
    sigma = module.factor_in.sigma
    optimizer = CSTPullbackAdam(
        nn.Sequential(module), target_step=0.01, cap=0.1
    )
    before = float(sigma.detach())
    x = torch.randn(4, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(3))
    optimizer.zero_grad()
    module(x).square().sum().backward()
    optimizer.step()
    moved = abs(float(sigma.detach()) - before)
    assert moved == pytest.approx(0.01 * before, rel=1e-2)
    assert moved <= 0.1 * before


def test_shared_sigma_steps_once_with_two_metric_terms():
    module, _store = _site(atoms=3, shared_factor=True)
    assert module.factor_in is module.factor_out
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    assert len(optimizer._sigmas) == 1
    (block,) = optimizer._sigmas
    assert len(block["users"]) == 2


def test_non_gaussian_family_raises():
    module, _store = _site(atoms=3)
    module.factor_in.family = "not-gaussian"  # instance shadow of the
    # ClassVar: the cheapest stand-in for a family without a closed form.
    with pytest.raises(ValueError, match="gaussian family only"):
        CSTPullbackAdam(nn.Sequential(module))
    CSTPullbackAdam(nn.Sequential(module), sigma_block=False)
