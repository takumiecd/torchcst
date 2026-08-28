"""CSTPullbackAdam's public ownership, geometry, and lifecycle contract."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from torchcst import CSTPullbackAdam, SampledKernelCoherence
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    MaturityGaussianFactor,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


def _birth(store, source, target, weights, *, start=0, extras=None):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(start, start + count, dtype=torch.int64),
        {} if extras is None else extras,
    )


def _site(
    *,
    atoms=4,
    capacity=None,
    name="pullback",
    normalized=True,
    learnable_mu=False,
    learnable_sigma=True,
    family="gaussian",
):
    capacity = atoms if capacity is None else capacity
    spec = RepresentationSpec.continuous(1, 1, factor=family)
    store = SynapseStore(name, 1, 1, capacity, spec=spec, dtype=torch.float64)
    generator = torch.Generator().manual_seed(17)
    extras = None
    if family == "maturity_gaussian":
        extras = {"maturity": torch.full((atoms, 1), -1.5, dtype=torch.float64)}
    store.apply(
        [
            _birth(
                store,
                torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
                torch.rand(atoms, 1, generator=generator, dtype=torch.float64),
                0.4 + torch.rand(atoms, generator=generator, dtype=torch.float64),
                extras=extras,
            )
        ]
    )
    mu_in = torch.linspace(-0.4, 1.4, 7, dtype=torch.float64).reshape(-1, 1)
    mu_out = torch.linspace(-0.3, 1.3, 6, dtype=torch.float64).reshape(-1, 1)
    inputs = NeuronStore(
        f"{name}-in",
        7,
        mu=nn.Parameter(mu_in) if learnable_mu else mu_in,
        initial_live=7,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        f"{name}-out",
        6,
        mu=mu_out,
        initial_live=6,
        dtype=torch.float64,
    )
    factor = (
        GaussianFactor(0.35, learnable=learnable_sigma)
        if family == "gaussian"
        else MaturityGaussianFactor(0.35, learnable=learnable_sigma)
    ).double()
    module = CSTLinear(
        inputs,
        outputs,
        store,
        factor,
        gauge=L2NormalizedColumns() if normalized else None,
    )
    return module, store


def _backward(module):
    x = torch.randn(
        3,
        module.in_features,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(29),
    )
    module(x).square().sum().backward()


def test_owns_every_cst_parameter_and_returns_the_dense_complement():
    site, store = _site(learnable_mu=True)
    dense = nn.Linear(site.out_features, 2).double()
    model = nn.Sequential(site, dense)
    optimizer = CSTPullbackAdam(model)
    owned = {id(parameter) for parameter in optimizer.owned_parameters()}
    for parameter in (
        store.w,
        store.s,
        store.t,
        site.in_neurons.mu,
        site.factor_in.sigma,
    ):
        assert id(parameter) in owned
    remainder = {id(parameter) for parameter in optimizer.non_cst_parameters()}
    assert id(dense.weight) in remainder and id(dense.bias) in remainder
    assert remainder == {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in owned
    }
    ordinary = torch.optim.AdamW(optimizer.non_cst_parameters(), lr=1e-3)
    optimizer.validate_dense_optimizer(ordinary)
    with pytest.raises(ValueError, match="also owns CST"):
        optimizer.validate_dense_optimizer(torch.optim.AdamW(model.parameters()))


def test_normalized_gauge_uses_exact_amplitude_orthogonality():
    module, _store = _site(normalized=True, learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    site = optimizer._atom_sites[0]
    slots = site.store.live_slots().to(site.store.w.device)
    gram = optimizer._atom_gram(site, slots)
    torch.testing.assert_close(gram[:, 0, 1:], torch.zeros_like(gram[:, 0, 1:]))
    torch.testing.assert_close(gram[:, 1:, 0], torch.zeros_like(gram[:, 1:, 0]))
    torch.testing.assert_close(gram[:, 0, 0], torch.ones_like(gram[:, 0, 0]))


def test_coordinate_cap_reads_per_atom_maturity_bandwidth():
    module, store = _site(family="maturity_gaussian", learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    site = optimizer._atom_sites[0]
    slots = store.live_slots()
    store.maturity.data[slots] = torch.tensor(
        [[-4.0], [-1.0], [1.0], [4.0]], dtype=torch.float64
    )

    expected = module.factor_in.effective_sigma(store.maturity[slots].reshape(-1))
    torch.testing.assert_close(
        optimizer._atom_effective_sigma(site, slots), expected
    )


@pytest.mark.parametrize(
    ("pullback", "pullback_scale"), [("inverse", 4.0), ("bounded", 1.0)]
)
def test_normalized_gaussian_split_whitener_matches_full_block(
    pullback, pullback_scale
):
    module, _store = _site(normalized=True, learnable_sigma=False)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        metric="block",
        pullback=pullback,
        pullback_scale=pullback_scale,
        subscribe=False,
    )
    site = optimizer._atom_sites[0]
    slots = site.store.live_slots().to(site.store.w.device)
    gram = optimizer._atom_gram(site, slots)
    value = torch.randn(
        slots.numel(),
        site.width,
        dtype=gram.dtype,
        generator=torch.Generator().manual_seed(41),
    )
    expected = optimizer._whitener(gram)(value)
    actual = optimizer._atom_whitener(site, gram)(value)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("metric", ["diag", "block"])
def test_shifted_metric_whitener_is_exact_and_bounded(metric):
    module, _store = _site(normalized=True, learnable_sigma=False)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        metric=metric,
        damping=0.0,
        pullback="bounded",
        pullback_scale=2.0,
        subscribe=False,
    )
    gram = torch.tensor([[[1e-8, 0.0], [0.0, 8.0]]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    whiten = optimizer._whitener(gram)
    actual = torch.stack(
        [whiten(identity[index : index + 1])[0] for index in range(2)], dim=1
    )
    expected = torch.diag(
        torch.tensor(
            [
                (1.0 + 1e-8 / 2.0) ** -0.5,
                (1.0 + 8.0 / 2.0) ** -0.5,
            ],
            dtype=torch.float64,
        )
    )
    torch.testing.assert_close(actual, expected)
    assert float(torch.linalg.matrix_norm(actual, ord=2)) <= 1.0


def test_public_pullback_defaults_preserve_inverse_compatibility():
    module, _store = _site(learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    assert optimizer.pullback == "inverse"
    assert optimizer.pullback_scale == 4.0
    assert optimizer.metric_shift is None
    assert "moment_space" not in optimizer.__dict__


@pytest.mark.parametrize("pullback", ["shifted", "tangent", None])
def test_pullback_must_name_a_supported_map(pullback):
    module, _store = _site(learnable_sigma=False)
    with pytest.raises(ValueError, match="pullback"):
        CSTPullbackAdam(
            nn.Sequential(module), pullback=pullback, subscribe=False
        )


@pytest.mark.parametrize("pullback_scale", [0.0, -1.0, True, None])
def test_pullback_scale_must_be_positive(pullback_scale):
    module, _store = _site(learnable_sigma=False)
    with pytest.raises(ValueError, match="pullback_scale"):
        CSTPullbackAdam(
            nn.Sequential(module),
            pullback="bounded",
            pullback_scale=pullback_scale,
            subscribe=False,
        )


def test_metric_shift_is_a_deprecated_compatibility_alias():
    module, _store = _site(learnable_sigma=False)
    with pytest.warns(DeprecationWarning, match="metric_shift is deprecated"):
        optimizer = CSTPullbackAdam(
            nn.Sequential(module), metric_shift=2.0, subscribe=False
        )
    assert optimizer.pullback == "bounded"
    assert optimizer.pullback_scale == 2.0
    assert optimizer.metric_shift == 2.0


@pytest.mark.parametrize("metric_shift", [0.0, -1.0, True])
def test_deprecated_metric_shift_still_validates_legacy_input(metric_shift):
    module, _store = _site(learnable_sigma=False)
    with pytest.raises(ValueError, match="metric_shift"):
        CSTPullbackAdam(
            nn.Sequential(module), metric_shift=metric_shift, subscribe=False
        )


def test_metric_shift_cannot_be_mixed_with_new_pullback_api():
    module, _store = _site(learnable_sigma=False)
    with pytest.raises(ValueError, match="cannot be combined"):
        CSTPullbackAdam(
            nn.Sequential(module),
            pullback="bounded",
            metric_shift=2.0,
            subscribe=False,
        )


def test_raw_gauge_keeps_amplitude_coordinate_correlation():
    module, _store = _site(normalized=False, learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    site = optimizer._atom_sites[0]
    gram = optimizer._atom_gram(site, site.store.live_slots().to(site.store.w.device))
    assert bool((gram[:, 0, 1:].abs() > 1e-10).any())
    torch.testing.assert_close(gram, gram.transpose(-1, -2))
    assert bool((torch.linalg.eigvalsh(gram) >= -1e-10).all())


@pytest.mark.parametrize("normalized", [False, True])
def test_same_atom_gram_matches_a_dense_autograd_oracle(normalized):
    module, store = _site(
        atoms=1,
        normalized=normalized,
        learnable_sigma=False,
    )
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    site = optimizer._atom_sites[0]
    actual = optimizer._atom_gram(site, torch.tensor([0]))[0]
    theta = torch.stack((store.w[0], store.s[0, 0], store.t[0, 0])).detach()

    def atom(values):
        source = values[1].reshape(1, 1)
        target = values[2].reshape(1, 1)
        incoming = module.gauge.columns(
            module.factor_in,
            module.in_neurons.mu,
            source,
        )[:, 0]
        outgoing = module.gauge.columns(
            module.factor_out,
            module.out_neurons.mu,
            target,
        )[:, 0]
        return values[0] * outgoing[:, None] * incoming[None, :]

    jacobian = torch.autograd.functional.jacobian(atom, theta).reshape(-1, 3)
    expected = jacobian.T @ jacobian
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_normalized_maturity_gram_matches_a_dense_autograd_oracle():
    module, store = _site(
        atoms=1,
        normalized=True,
        learnable_sigma=False,
        family="maturity_gaussian",
    )
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    site = optimizer._atom_sites[0]
    actual = optimizer._atom_gram(site, torch.tensor([0]))[0]
    theta = torch.stack(
        (store.w[0], store.s[0, 0], store.t[0, 0], store.maturity[0, 0])
    ).detach()

    def atom(values):
        source = values[1].reshape(1, 1)
        target = values[2].reshape(1, 1)
        extras = {"maturity": values[3].reshape(1, 1)}
        incoming = module.gauge.columns(
            module.factor_in, module.in_neurons.mu, source, extras
        )[:, 0]
        outgoing = module.gauge.columns(
            module.factor_out, module.out_neurons.mu, target, extras
        )[:, 0]
        return values[0] * outgoing[:, None] * incoming[None, :]

    jacobian = torch.autograd.functional.jacobian(atom, theta).reshape(-1, 4)
    expected = jacobian.T @ jacobian
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("metric", ["diag", "block"])
@pytest.mark.parametrize("normalized", [False, True])
def test_direct_lr_steps_all_core_blocks(metric, normalized):
    module, store = _site(normalized=normalized, learnable_mu=True)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        metric=metric,
        lr=2e-3,
        max_step_sigma=None,
    )
    before = {
        "w": store.w.detach().clone(),
        "s": store.s.detach().clone(),
        "t": store.t.detach().clone(),
        "mu": module.in_neurons.mu.detach().clone(),
        "sigma": module.factor_in.sigma.detach().clone(),
    }
    optimizer.zero_grad(set_to_none=True)
    _backward(module)
    optimizer.step()
    current = {
        "w": store.w,
        "s": store.s,
        "t": store.t,
        "mu": module.in_neurons.mu,
        "sigma": module.factor_in.sigma,
    }
    for name, value in before.items():
        assert not torch.equal(current[name].detach(), value), name


def test_lr_and_target_map_step_are_exclusive():
    module, _store = _site(learnable_sigma=False)
    with pytest.raises(ValueError, match="exactly one"):
        CSTPullbackAdam(nn.Sequential(module), lr=1e-3, target_map_step=0.01)
    with pytest.raises(ValueError, match="exactly one"):
        CSTPullbackAdam(nn.Sequential(module), lr=None, target_map_step=None)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        lr=None,
        target_map_step=0.01,
        max_step_sigma=None,
    )
    optimizer.zero_grad()
    _backward(module)
    optimizer.step()
    assert optimizer.param_groups[0]["calibrated_scale"] is not None


def test_kernel_coherence_is_opt_in_and_reduces_exact_alignment() -> None:
    module, store = _site(atoms=5, learnable_sigma=False)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        lr=1e-2,
        betas=(0.0, 0.0),
        max_step_sigma=0.1,
        coherence=SampledKernelCoherence(0.1, pairs=None),
        coherence_seed=13,
        subscribe=False,
    )

    def exact_value():
        slots = store.live_slots().to(store.s.device)
        _, _, value = optimizer._coherence_gradient(optimizer._atom_sites[0], slots, 0)
        return float(value)

    before = exact_value()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        store.w.grad = torch.zeros_like(store.w)
        store.s.grad = torch.zeros_like(store.s)
        store.t.grad = torch.zeros_like(store.t)
        optimizer.step()
    after = exact_value()

    assert optimizer.last_coherence is not None
    assert after < 0.8 * before

    plain_module, _ = _site(atoms=5, name="plain-no-coherence", learnable_sigma=False)
    plain = CSTPullbackAdam(nn.Sequential(plain_module), subscribe=False)
    assert plain.coherence is None and plain.last_coherence is None


def test_decoupled_kernel_coherence_moves_outside_adam_moments() -> None:
    module, store = _site(atoms=5, learnable_sigma=False)
    optimizer = CSTPullbackAdam(
        nn.Sequential(module),
        lr=1e-2,
        betas=(0.0, 0.0),
        max_step_sigma=0.1,
        decoupled_coherence=SampledKernelCoherence(1.0, pairs=None),
        coherence_lr=0.05,
        subscribe=False,
    )
    before_s = store.s.detach().clone()
    before_t = store.t.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    store.w.grad = torch.zeros_like(store.w)
    store.s.grad = torch.zeros_like(store.s)
    store.t.grad = torch.zeros_like(store.t)
    optimizer.step()

    state = optimizer._atom_sites[0].state
    assert torch.count_nonzero(state["m"]) == 0
    assert torch.count_nonzero(state["v"]) == 0
    assert not torch.equal(store.s, before_s)
    assert not torch.equal(store.t, before_t)
    assert optimizer.last_coherence is not None


def test_coupled_and_decoupled_kernel_coherence_are_exclusive() -> None:
    module, _ = _site(atoms=3, learnable_sigma=False)
    force = SampledKernelCoherence(1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CSTPullbackAdam(
            nn.Sequential(module),
            coherence=force,
            decoupled_coherence=force,
            coherence_lr=0.1,
            subscribe=False,
        )
    with pytest.raises(ValueError, match="required exactly"):
        CSTPullbackAdam(
            nn.Sequential(module),
            decoupled_coherence=force,
            subscribe=False,
        )


def test_sigma_cap_is_independent_and_can_be_disabled():
    module, store = _site(learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module), lr=1.0, max_step_sigma=0.02)
    before_s, before_t = store.s.detach().clone(), store.t.detach().clone()
    optimizer.zero_grad()
    _backward(module)
    optimizer.step()
    movement = (
        (store.s.detach() - before_s).square().sum(1)
        + (store.t.detach() - before_t).square().sum(1)
    ).sqrt()
    assert float(movement.max()) <= 0.02 * 0.35 * (1 + 1e-8)


def test_sigma_metric_uses_scalar_forward_mode(monkeypatch):
    module, _store = _site(learnable_sigma=True)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)

    def reject_reverse_jacobian(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scalar sigma must not build a reverse-mode Jacobian")

    monkeypatch.setattr(torch.autograd.functional, "jacobian", reject_reverse_jacobian)
    metric = optimizer._sigma_metric(optimizer._sigmas[0])
    assert metric.ndim == 0
    assert bool(torch.isfinite(metric))
    assert float(metric) > 0


def test_normalized_gaussian_sigma_metric_matches_same_atom_oracle():
    module, store = _site(atoms=2, learnable_sigma=True, normalized=True)
    optimizer = CSTPullbackAdam(nn.Sequential(module), subscribe=False)
    actual = optimizer._sigma_metric(optimizer._sigmas[0])
    slots = store.live_slots()
    sigma = module.factor_in.sigma.detach().clone().requires_grad_(True)
    expected = sigma.new_zeros(())

    for slot in slots.tolist():
        source = store.s.detach()[slot : slot + 1]
        target = store.t.detach()[slot : slot + 1]
        weight = store.w.detach()[slot]

        def atom(value, local_source=source, local_target=target, local_weight=weight):
            def unit(query, center):
                distance = torch.cdist(query, center).square()
                raw = torch.exp(-distance / (2.0 * value.square()))
                return raw[:, 0] / torch.linalg.vector_norm(raw[:, 0])

            incoming = unit(module.in_neurons.mu, local_source)
            outgoing = unit(module.out_neurons.mu, local_target)
            return local_weight * outgoing[:, None] * incoming[None, :]

        derivative = torch.autograd.functional.jacobian(atom, sigma)
        expected = expected + derivative.square().sum()

    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_factor_declared_columns_join_the_same_atom_block():
    module, store = _site(
        family="maturity_gaussian",
        learnable_sigma=False,
        normalized=True,
    )
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    site = optimizer._atom_sites[0]
    assert [field.name for field in site.fields] == ["w", "s", "t", "maturity"]
    before = store.maturity.detach().clone()
    optimizer.zero_grad()
    _backward(module)
    optimizer.step()
    assert not torch.equal(store.maturity.detach(), before)


def test_structural_death_and_birth_reset_reused_slot_state():
    module, store = _site(capacity=5, atoms=4, learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    optimizer.zero_grad()
    _backward(module)
    optimizer.step()
    state = optimizer._atom_sites[0].state
    victim = store.live_ids()[:1]
    victim_slot = store._slots.slots_of(victim)
    assert bool((state["m"].index_select(0, victim_slot) != 0).any())
    store.apply([SynapseDeath(store.site, victim)])
    assert torch.equal(
        state["m"].index_select(0, victim_slot),
        torch.zeros_like(state["m"].index_select(0, victim_slot)),
    )
    store.apply([_birth(store, [[0.2]], [[0.8]], [0.5], start=99)])
    assert torch.equal(
        state["m"].index_select(0, victim_slot),
        torch.zeros_like(state["m"].index_select(0, victim_slot)),
    )


def test_optimizer_state_dict_roundtrip_includes_custom_moments():
    module, _store = _site(learnable_mu=True)
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    optimizer.zero_grad()
    _backward(module)
    optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    kept = optimizer._atom_sites[0].state["m"].clone()
    optimizer._atom_sites[0].state["m"].zero_()
    optimizer.load_state_dict(saved)
    torch.testing.assert_close(optimizer._atom_sites[0].state["m"], kept)


def test_is_a_standard_torch_optimizer():
    module, _store = _site(learnable_sigma=False)
    optimizer = CSTPullbackAdam(nn.Sequential(module))
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.param_groups
