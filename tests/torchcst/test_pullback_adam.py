"""PullbackAdam's moment-space semantics and lifecycle contract."""

from __future__ import annotations

import pytest
import torch

from torchcst import PullbackAdam
from torchcst.compute import CSTLinear
from torchcst.optim.pullback import (
    _gaussian_l2_metric_block,
    _gaussian_l2_metric_gram,
)
from torchcst.representation import (
    GaussianKernel,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import (
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseStore,
)


def _birth(store, source, target, weights, lineage_start=0):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def _site(*, atoms=2, capacity=None, d_in=1, d_out=1, normalized=True):
    capacity = atoms if capacity is None else capacity
    store = SynapseStore(
        "pullback-test",
        d_in,
        d_out,
        capacity,
        spec=RepresentationSpec.continuous(d_in, d_out),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(41)
    source = torch.rand(atoms, d_in, generator=generator, dtype=torch.float64)
    target = torch.rand(atoms, d_out, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    inputs = NeuronStore(
        "pullback-input",
        7,
        mu=torch.linspace(-0.5, 1.5, 7, dtype=torch.float64)
        .reshape(-1, 1)
        .expand(-1, d_in),
        initial_live=7,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "pullback-output",
        6,
        mu=torch.linspace(-0.4, 1.4, 6, dtype=torch.float64)
        .reshape(-1, 1)
        .expand(-1, d_out),
        initial_live=6,
        dtype=torch.float64,
    )
    gauge = L2NormalizedColumns() if normalized else None
    module = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianKernel(0.35).double(),
        gauge=gauge,
    )
    return module, store


def _scattered_site(*, atoms=3, d_in=2, d_out=2, seed=7):
    """A multi-axis site whose neuron cloud is irregular.

    ``_site`` replicates one 1-D grid across every chart axis, which makes
    all axis derivatives identical; scattering the neurons is what gives the
    within-atom axis Gram a non-trivial off-diagonal to test.
    """
    store = SynapseStore(
        "pullback-block-test",
        d_in,
        d_out,
        atoms,
        spec=RepresentationSpec.continuous(d_in, d_out),
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    source = torch.rand(atoms, d_in, generator=generator, dtype=torch.float64)
    target = torch.rand(atoms, d_out, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    inputs = NeuronStore(
        "pullback-block-input",
        9,
        mu=torch.rand(9, d_in, generator=generator, dtype=torch.float64)
        * 2.0
        - 0.5,
        initial_live=9,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "pullback-block-output",
        8,
        mu=torch.rand(8, d_out, generator=generator, dtype=torch.float64)
        * 2.0
        - 0.5,
        initial_live=8,
        dtype=torch.float64,
    )
    module = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianKernel(0.35).double(),
        gauge=L2NormalizedColumns(),
    )
    return module, store


def _constant_metric(store):
    diagonal_s = torch.linspace(
        0.5, 1.5, store.s.numel(), dtype=store.s.dtype
    ).reshape_as(store.s)
    diagonal_t = torch.linspace(
        0.7, 1.7, store.t.numel(), dtype=store.t.dtype
    ).reshape_as(store.t)
    return diagonal_s, diagonal_t


def test_parameter_and_tangent_moments_agree_when_metric_is_fixed() -> None:
    parameter_module, parameter_store = _site(d_in=2)
    tangent_module, tangent_store = _site(d_in=2)
    parameter = PullbackAdam(
        parameter_module,
        moment_space="parameter",
        eps=0.0,
        damping=0.0,
        cap_sigma=100.0,
    )
    tangent = PullbackAdam(
        tangent_module,
        moment_space="tangent",
        eps=0.0,
        damping=0.0,
        cap_sigma=100.0,
    )
    fixed = _constant_metric(parameter_store)
    parameter._metric_diag = lambda live: tuple(x.clone() for x in fixed)
    tangent._metric_diag = lambda live: tuple(x.clone() for x in fixed)

    generator = torch.Generator().manual_seed(9)
    for _ in range(4):
        grad_s = torch.randn(
            parameter_store.s.shape, generator=generator, dtype=torch.float64
        )
        grad_t = torch.randn(
            parameter_store.t.shape, generator=generator, dtype=torch.float64
        )
        parameter_store.s.grad = grad_s.clone()
        parameter_store.t.grad = grad_t.clone()
        tangent_store.s.grad = grad_s.clone()
        tangent_store.t.grad = grad_t.clone()
        parameter.step()
        tangent.step()

    torch.testing.assert_close(
        parameter_store.s, tangent_store.s, rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(
        parameter_store.t, tangent_store.t, rtol=1e-11, atol=1e-12
    )


def test_parameter_and_tangent_moments_separate_when_metric_moves() -> None:
    parameter_module, parameter_store = _site()
    tangent_module, tangent_store = _site()
    parameter = PullbackAdam(
        parameter_module,
        moment_space="parameter",
        eps=0.0,
        damping=0.0,
        cap_sigma=100.0,
    )
    tangent = PullbackAdam(
        tangent_module,
        moment_space="tangent",
        eps=0.0,
        damping=0.0,
        cap_sigma=100.0,
    )
    sequence = [
        (
            torch.full_like(parameter_store.s, 0.25),
            torch.full_like(parameter_store.t, 4.0),
        ),
        (
            torch.full_like(parameter_store.s, 4.0),
            torch.full_like(parameter_store.t, 0.25),
        ),
    ]
    for index, diagonal in enumerate(sequence):
        parameter._metric_diag = (
            lambda live, diagonal=diagonal: tuple(
                value.clone() for value in diagonal
            )
        )
        tangent._metric_diag = (
            lambda live, diagonal=diagonal: tuple(
                value.clone() for value in diagonal
            )
        )
        grad_s = torch.full_like(parameter_store.s, 0.3 + index)
        grad_t = torch.full_like(parameter_store.t, -0.8 + 0.2 * index)
        parameter_store.s.grad = grad_s.clone()
        parameter_store.t.grad = grad_t.clone()
        tangent_store.s.grad = grad_s.clone()
        tangent_store.t.grad = grad_t.clone()
        parameter.step()
        tangent.step()

    assert not torch.allclose(parameter_store.s, tangent_store.s)
    assert not torch.allclose(parameter_store.t, tangent_store.t)


def test_diagonal_metric_predicts_a_one_atom_weight_displacement() -> None:
    module, store = _site(atoms=1)
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1
    )
    live = store.live_slots().to(store.s.device)
    diagonal_s, diagonal_t = optimizer._metric_diag(live)
    delta_s = torch.tensor([[2.0e-6]], dtype=store.s.dtype)
    delta_t = torch.tensor([[-1.5e-6]], dtype=store.t.dtype)
    before = module.dense_weight().detach()
    with torch.no_grad():
        store.s.add_(delta_s)
        store.t.add_(delta_t)
    after = module.dense_weight().detach()

    actual = (after - before).square().sum()
    predicted = (
        (diagonal_s * delta_s.square()).sum()
        + (diagonal_t * delta_t.square()).sum()
    )
    torch.testing.assert_close(actual, predicted, rtol=2e-5, atol=1e-18)


def test_fused_gaussian_formula_matches_the_generic_metric() -> None:
    module, store = _site(atoms=2, d_in=2)
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1
    )
    expected = optimizer._metric_diag(store.live_slots())
    actual = _gaussian_l2_metric_block(
        module.in_neurons.mu,
        module.out_neurons.mu,
        store.s,
        store.t,
        store.w.square(),
        module.kernel_in.sigma,
        module.kernel_out.sigma,
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


def test_block_metric_diagonal_matches_the_per_axis_diagonal() -> None:
    module, store = _scattered_site()
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1, metric="block"
    )
    live = store.live_slots()
    gram_s, gram_t = optimizer._metric_gram(live)
    diagonal_s, diagonal_t = optimizer._metric_diag(live)
    torch.testing.assert_close(
        gram_s.diagonal(dim1=-2, dim2=-1), diagonal_s
    )
    torch.testing.assert_close(
        gram_t.diagonal(dim1=-2, dim2=-1), diagonal_t
    )


def test_block_metric_predicts_a_mixed_axis_weight_displacement() -> None:
    module, store = _scattered_site(atoms=1)
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1, metric="block"
    )
    live = store.live_slots()
    gram_s, gram_t = optimizer._metric_gram(live)
    coupling = gram_s[0, 0, 1].abs() / gram_s[0].diagonal().mean()
    assert coupling > 0.05  # the fixture must exercise the off-diagonal

    delta_s = torch.tensor([[2.0e-6, -1.4e-6]], dtype=store.s.dtype)
    delta_t = torch.tensor([[-1.5e-6, 0.9e-6]], dtype=store.t.dtype)
    before = module.dense_weight().detach()
    with torch.no_grad():
        store.s.add_(delta_s)
        store.t.add_(delta_t)
    after = module.dense_weight().detach()

    actual = (after - before).square().sum()
    predicted = (
        delta_s[0] @ gram_s[0] @ delta_s[0]
        + delta_t[0] @ gram_t[0] @ delta_t[0]
    )
    torch.testing.assert_close(actual, predicted, rtol=2e-5, atol=1e-18)

    diagonal_only = (
        gram_s[0].diagonal() @ delta_s[0].square()
        + gram_t[0].diagonal() @ delta_t[0].square()
    )
    assert (diagonal_only - actual).abs() > 10.0 * (predicted - actual).abs()


def test_fused_gaussian_gram_matches_the_generic_gram() -> None:
    module, store = _scattered_site(atoms=2)
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1, metric="block"
    )
    expected = optimizer._metric_gram(store.live_slots())
    actual = _gaussian_l2_metric_gram(
        module.in_neurons.mu,
        module.out_neurons.mu,
        store.s,
        store.t,
        store.w.square(),
        module.kernel_in.sigma,
        module.kernel_out.sigma,
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)


def test_block_metric_steps_deterministically_and_follows_lifecycle() -> None:
    runs = []
    for _ in range(2):
        module, store = _scattered_site()
        optimizer = PullbackAdam(
            module, moment_space="tangent", cap_sigma=0.1, metric="block"
        )
        generator = torch.Generator().manual_seed(3)
        for _ in range(3):
            store.s.grad = torch.randn(
                store.s.shape, generator=generator, dtype=torch.float64
            )
            store.t.grad = torch.randn(
                store.t.shape, generator=generator, dtype=torch.float64
            )
            optimizer.step()
        runs.append((store.s.detach().clone(), store.t.detach().clone()))
    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(runs[0][1], runs[1][1], rtol=0, atol=0)

    module, store = _scattered_site()
    optimizer = PullbackAdam(
        module, moment_space="tangent", cap_sigma=0.1, metric="block"
    )
    store.s.grad = torch.full_like(store.s, 0.3)
    store.t.grad = torch.full_like(store.t, -0.2)
    optimizer.step()
    victim = store.live_ids()[:1]
    store.apply(
        [
            SynapseDeath(store.site, victim),
            _birth(
                store,
                [[0.5, 0.5]],
                [[0.5, 0.5]],
                [1.0],
                lineage_start=10,
            ),
        ]
    )
    for name in ("m_s", "m_t", "v_s", "v_t", "travel"):
        assert torch.count_nonzero(getattr(optimizer, name)[0]) == 0
    store.s.grad = torch.full_like(store.s, 0.1)
    store.t.grad = torch.full_like(store.t, 0.1)
    optimizer.step()


def test_block_metric_state_rejects_a_diag_optimizer() -> None:
    module, store = _scattered_site()
    optimizer = PullbackAdam(
        module, moment_space="tangent", cap_sigma=0.1, metric="block"
    )
    store.s.grad = torch.full_like(store.s, 0.3)
    store.t.grad = torch.full_like(store.t, -0.2)
    optimizer.step()
    state = optimizer.state_dict()
    assert state["metric"] == "block"

    twin_module, _ = _scattered_site()
    restored = PullbackAdam(
        twin_module, moment_space="tangent", cap_sigma=0.1, metric="block"
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.m_s, optimizer.m_s)

    diag_module, _ = _scattered_site()
    diag = PullbackAdam(diag_module, moment_space="tangent", cap_sigma=0.1)
    with pytest.raises(ValueError, match="metric"):
        diag.load_state_dict(state)

    legacy = dict(state)
    del legacy["metric"]  # checkpoints written before the option existed
    with pytest.raises(ValueError, match="metric"):
        restored.load_state_dict(legacy)


def test_state_is_coordinate_shaped_and_checkpoint_roundtrips() -> None:
    module, store = _site(atoms=3, d_in=2)
    optimizer = PullbackAdam(
        module, moment_space="tangent", cap_sigma=0.1
    )
    store.s.grad = torch.full_like(store.s, 0.3)
    store.t.grad = torch.full_like(store.t, -0.2)
    optimizer.step()
    state = optimizer.state_dict()
    shapes = {
        name: value.shape
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    assert shapes == {
        "m_s": store.s.shape,
        "m_t": store.t.shape,
        "v_s": store.s.shape,
        "v_t": store.t.shape,
        "travel": (store.capacity,),
    }
    assert (module.out_features, module.in_features) not in shapes.values()

    twin_module, _ = _site(atoms=3, d_in=2)
    restored = PullbackAdam(
        twin_module, moment_space="tangent", cap_sigma=0.1
    )
    restored.load_state_dict(state)
    assert restored.step_count == optimizer.step_count
    assert restored.eta == optimizer.eta
    for name in ("m_s", "m_t", "v_s", "v_t", "travel"):
        torch.testing.assert_close(getattr(restored, name), getattr(optimizer, name))


def test_follower_zeros_reused_slots_and_grows_every_moment() -> None:
    module, store = _site(atoms=3, capacity=4)
    optimizer = PullbackAdam(
        module, moment_space="parameter", cap_sigma=0.1
    )
    store.s.grad = torch.full_like(store.s, 0.3)
    store.t.grad = torch.full_like(store.t, -0.2)
    optimizer.step()
    victim = store.live_ids()[:1]
    store.apply(
        [
            SynapseDeath(store.site, victim),
            _birth(store, [[0.5]], [[0.5]], [1.0], lineage_start=10),
        ]
    )
    for name in ("m_s", "m_t", "v_s", "v_t", "travel"):
        assert torch.count_nonzero(getattr(optimizer, name)[0]) == 0

    store.apply(
        [
            _birth(
                store,
                torch.full((4, 1), 0.25),
                torch.full((4, 1), 0.75),
                [1.0] * 4,
                lineage_start=20,
            )
        ]
    )
    for name in ("m_s", "m_t", "v_s", "v_t", "travel"):
        assert getattr(optimizer, name).shape[0] == store.capacity


def test_rejects_ambiguous_or_unsupported_configurations() -> None:
    module, _ = _site()
    raw_module, _ = _site(normalized=False)
    with pytest.raises(ValueError, match="moment_space"):
        PullbackAdam(module, moment_space="current", cap_sigma=0.1)
    with pytest.raises(ValueError, match="metric"):
        PullbackAdam(
            module, moment_space="parameter", cap_sigma=0.1, metric="full"
        )
    with pytest.raises(TypeError, match="L2NormalizedColumns"):
        PullbackAdam(raw_module, moment_space="parameter", cap_sigma=0.1)
    with pytest.raises(ValueError, match="betas"):
        PullbackAdam(
            module,
            moment_space="parameter",
            cap_sigma=0.1,
            betas=(0.9, 1.0),
        )
