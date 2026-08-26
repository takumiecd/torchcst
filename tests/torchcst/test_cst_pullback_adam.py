"""CSTPullbackAdam: one class, one template, every CST parameter."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchcst import (
    CSTPullbackAdam,
    PullbackAdam,
    amplitude_leaf,
    install_whitened_basis,
)
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import (
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def _birth(store, source, target, weights):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(count, dtype=torch.int64),
    )


def _site(*, atoms=5, seed=43, name="single-test", normalized=True,
          learnable_mu=False):
    store = SynapseStore(
        name,
        1,
        1,
        atoms,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    store.apply([_birth(
        store,
        torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
        torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
        0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64),
    )])
    mu_in = torch.linspace(-0.5, 1.5, 7, dtype=torch.float64).reshape(-1, 1)
    inputs = NeuronStore(
        f"{name}-input",
        7,
        mu=nn.Parameter(mu_in) if learnable_mu else mu_in,
        initial_live=7,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        f"{name}-output",
        6,
        mu=torch.linspace(-0.4, 1.4, 6, dtype=torch.float64).reshape(-1, 1),
        initial_live=6,
        dtype=torch.float64,
    )
    module = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianFactor(0.35).double(),
        gauge=L2NormalizedColumns() if normalized else None,
    )
    return module, store


def test_coordinate_steps_match_pullback_adam_exactly():
    """The uniformity claim, verified: the single class's coordinate block
    reproduces PullbackAdam (diag metric, tangent moments) bit for bit."""
    dials = {"target_step": 0.01, "betas": (0.9, 0.999), "eps": 1e-8,
             "damping": 1e-2}
    module_a, store_a = _site(name="twin-a")
    module_b, store_b = _site(name="twin-b")
    twin = PullbackAdam(
        module_a,
        moment_space="tangent",
        metric="diag",
        cap_sigma=0.1,
        target_step=dials["target_step"],
        betas=dials["betas"],
        eps=dials["eps"],
        damping=dials["damping"],
    )
    single = CSTPullbackAdam(
        nn.Sequential(module_b), cap=0.1, lr_w=0.0, sigma_block=False,
        **dials,
    )
    x = torch.randn(4, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(47))
    for _ in range(3):
        for module in (module_a, module_b):
            module.zero_grad(set_to_none=True)
            module(x).square().sum().backward()
        twin.step()
        single.step()
        torch.testing.assert_close(store_b.s.detach(), store_a.s.detach())
        torch.testing.assert_close(store_b.t.detach(), store_a.t.detach())
        torch.testing.assert_close(store_b.w.detach(), store_a.w.detach())


def test_amplitude_block_matches_the_parametrized_route():
    lr, wd = 1e-2, 0.1
    module_a, store_a = _site(name="amp-a")
    module_b, _store_b = _site(name="amp-b")
    install_whitened_basis(module_a, ridge=1e-2)
    adamw = torch.optim.AdamW(
        [amplitude_leaf(store_a)], lr=lr, weight_decay=wd
    )
    single = CSTPullbackAdam(
        nn.Sequential(module_b), lr_w=lr, weight_decay_w=wd, w_ridge=1e-2,
    )
    x = torch.randn(4, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(11))
    for _ in range(4):
        adamw.zero_grad(set_to_none=True)
        module_a(x).square().sum().backward()
        adamw.step()

        single.zero_grad()
        module_b(x).square().sum().backward()
        single._step_amplitudes(single._sites[0], 1.0)

        torch.testing.assert_close(
            module_b.dense_weight().detach(),
            module_a.dense_weight().detach(),
            rtol=1e-8,
            atol=1e-10,
        )


def test_full_step_moves_every_cst_block_and_nothing_dense():
    module, store = _site(learnable_mu=True)
    dense = nn.Linear(6, 2).double()
    model = nn.Sequential(module, nn.Flatten(), dense)
    optimizer = CSTPullbackAdam(model)
    before = {
        "s": store.s.detach().clone(),
        "t": store.t.detach().clone(),
        "w": store.w.detach().clone(),
        "mu": module.in_neurons.mu.detach().clone(),
        "sigma": module.factor_in.sigma.detach().clone(),
        "dense": dense.weight.detach().clone(),
    }
    x = torch.randn(4, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(53))
    optimizer.zero_grad()
    model(x).square().sum().backward()
    optimizer.step()
    assert not torch.equal(store.s.detach(), before["s"])
    assert not torch.equal(store.t.detach(), before["t"])
    assert not torch.equal(store.w.detach(), before["w"])
    assert not torch.equal(module.in_neurons.mu.detach(), before["mu"])
    assert not torch.equal(module.factor_in.sigma.detach(), before["sigma"])
    assert torch.equal(dense.weight.detach(), before["dense"])


def test_requires_the_l2_gauge():
    module, _store = _site(normalized=False)
    with pytest.raises(TypeError, match="L2-normalised"):
        CSTPullbackAdam(nn.Sequential(module))


def test_structural_event_raises_at_step():
    module, store = _site()
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    store.apply([SynapseDeath(store.site, store.live_ids()[:1])])
    x = torch.randn(2, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(59))
    optimizer.zero_grad()
    module(x).square().sum().backward()
    with pytest.raises(RuntimeError, match="version"):
        optimizer.step()


def test_state_dict_roundtrip():
    module, _store = _site(learnable_mu=True)
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    x = torch.randn(2, 7, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(61))
    optimizer.zero_grad()
    module(x).square().sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()
    kept = optimizer._sites[0]["coord"]["m"][0].clone()
    optimizer._sites[0]["coord"]["m"][0].zero_()
    optimizer._charts[0]["m"][0].zero_()
    optimizer.load_state_dict(saved)
    torch.testing.assert_close(optimizer._sites[0]["coord"]["m"][0], kept)
    assert optimizer._sites[0]["coord"]["step"] == 1
    assert optimizer._charts[0]["step"] == 1
