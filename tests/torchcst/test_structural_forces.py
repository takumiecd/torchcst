"""Rent, repulsion, and PullbackAdam's continuous-lifecycle contract."""

from __future__ import annotations

import pytest
import torch

from torchcst import PairRepulsion, PullbackAdam, SmoothRent
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.35


def _line_site(source, target, weights, *, name="forces-test"):
    source = torch.as_tensor(source, dtype=torch.float64).reshape(-1, 1)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(-1, 1)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    atoms = weights.numel()
    store = SynapseStore(
        name,
        1,
        1,
        atoms,
        spec=RepresentationSpec.continuous(1, 1),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                source,
                target,
                weights,
                torch.arange(atoms, dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        f"{name}-in",
        21,
        mu=torch.linspace(-0.5, 1.5, 21, dtype=torch.float64).reshape(-1, 1),
        initial_live=21,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        f"{name}-out",
        19,
        mu=torch.linspace(-0.5, 1.5, 19, dtype=torch.float64).reshape(-1, 1),
        initial_live=19,
        dtype=torch.float64,
    )
    module = CSTLinear(
        inputs,
        outputs,
        store,
        GaussianFactor(SIGMA).double(),
        gauge=L2NormalizedColumns(),
    )
    return module, store


def _target_weight(placements):
    module, _ = _line_site(
        [p[0] for p in placements],
        [p[1] for p in placements],
        [p[2] for p in placements],
        name="forces-target",
    )
    return module.dense_weight().detach()


def _train(module, store, optimizer, w_target, steps):
    for _ in range(steps):
        optimizer.zero_grad()
        loss = 0.5 * (module.dense_weight() - w_target).square().sum()
        loss.backward()
        optimizer.step()


def test_smooth_rent_gradient_and_schedule() -> None:
    rent = SmoothRent(0.3, eps=0.1)
    w = torch.tensor([1.0, -0.5, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        rent.gradient(w, 0), 0.3 * w / (w * w + 0.01).sqrt()
    )
    scheduled = SmoothRent(lambda step: 0.5 if step < 10 else 0.1, eps=0.1)
    assert scheduled.rate(0) == 0.5
    assert scheduled.rate(11) == 0.1
    with pytest.raises(ValueError, match="lam"):
        SmoothRent(-0.1)
    with pytest.raises(ValueError, match="eps"):
        SmoothRent(0.1, eps=0.0)


def test_pair_repulsion_pushes_overlapping_atoms_apart() -> None:
    repulsion = PairRepulsion(0.5, pairs=1 << 20)
    source = torch.tensor(
        [[0.50], [0.55], [2.50]], dtype=torch.float64
    )
    target = torch.tensor(
        [[0.50], [0.50], [-2.50]], dtype=torch.float64
    )
    sigma = torch.tensor(SIGMA, dtype=torch.float64)
    g_s, _ = repulsion.gradient(
        source, target, torch.arange(3), sigma, sigma, None
    )
    # descent moves atom 0 down and atom 1 up: apart along s
    assert g_s[0, 0] > 0 and g_s[1, 0] < 0
    torch.testing.assert_close(
        g_s.sum(), torch.tensor(0.0, dtype=torch.float64),
        rtol=0, atol=1e-12,
    )
    assert g_s[2, 0].abs() < 1e-8  # the far atom feels nothing
    dead_s, _ = repulsion.gradient(
        source, target, torch.tensor([0, 1]), sigma, sigma, None
    )
    assert dead_s[2].abs().max() == 0  # rows outside `live` stay zero


def test_log_barrier_force_diverges_toward_identical_columns() -> None:
    sigma = torch.tensor(1.0, dtype=torch.float64)

    def force_at(distance):
        source = torch.tensor([[0.0], [distance]], dtype=torch.float64)
        target = torch.zeros_like(source)
        force = PairRepulsion(
            1.0, pairs=1 << 20, potential="log_barrier"
        )
        gradient, _ = force.gradient(
            source, target, torch.arange(2), sigma, sigma, None
        )
        return gradient[0, 0].abs()

    near = force_at(1e-3)
    far = force_at(0.1)
    assert near > 50 * far
    assert near > 500  # asymptotically 2 / r
    with pytest.raises(ValueError, match="potential"):
        PairRepulsion(1.0, potential="inverse-quartic")


def test_pair_repulsion_sampling_is_deterministic_and_aligned() -> None:
    generator = torch.Generator().manual_seed(5)
    source = torch.rand(24, 1, generator=generator, dtype=torch.float64)
    target = torch.rand(24, 1, generator=generator, dtype=torch.float64)
    sigma = torch.tensor(SIGMA, dtype=torch.float64)
    live = torch.arange(24)
    exact_s, _ = PairRepulsion(0.5, pairs=1 << 20).gradient(
        source, target, live, sigma, sigma, None
    )
    sampled = PairRepulsion(0.5, pairs=64)

    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return sampled.gradient(source, target, live, sigma, sigma, g)

    once_s, _ = draw(7)
    again_s, _ = draw(7)
    torch.testing.assert_close(once_s, again_s, rtol=0, atol=0)
    other_s, _ = draw(8)
    assert not torch.equal(once_s, other_s)

    mean_s = torch.stack(
        [draw(seed)[0] for seed in range(200)]
    ).mean(0)
    cosine = torch.nn.functional.cosine_similarity(
        mean_s.flatten(), exact_s.flatten(), dim=0
    )
    assert cosine > 0.95  # unbiased estimator points the same way


def test_rent_grows_profitable_and_starves_stranded_atoms() -> None:
    w_target = _target_weight([(0.2, 0.2, 1.0)])
    module, store = _line_site([0.2, 1.0], [0.2, 1.0], [0.5, 0.4])
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=0.05,
        target_step=0.02,
        rent=SmoothRent(0.05, eps=1e-3),
        lr_w=0.03,
    )
    _train(module, store, optimizer, w_target, steps=300)
    assert store.w[0].item() > 0.7        # profitable atom kept growing
    assert abs(store.w[1].item()) < 0.05  # stranded atom starved to ~0


def test_ghost_migrates_up_the_score_field_and_resurrects() -> None:
    w_target = _target_weight([(0.15, 0.15, 1.0), (0.95, 0.95, 0.8)])
    module, store = _line_site(
        [0.15, 0.55], [0.15, 0.55], [1.0, 1e-4]
    )
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=0.2,
        target_step=0.05,
        rent=SmoothRent(0.1, eps=1e-3),
        lr_w=0.03,
    )
    _train(module, store, optimizer, w_target, steps=800)
    # the anchored live atom stays put on its feature
    assert (store.s[0] - 0.15).abs().item() < 0.05
    assert store.w[0].item() > 0.8
    # the ghost walked ~1.6 sigma per side to the empty feature and woke
    assert (store.s[1] - 0.95).abs().item() < 0.15
    assert (store.t[1] - 0.95).abs().item() < 0.15
    assert store.w[1].item() > 0.3


def test_wall_clamps_wandering_coordinates() -> None:
    module, store = _line_site([0.95], [0.95], [0.5])
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=5.0,
        target_step=1.0,
        wall=True,
    )
    for _ in range(20):
        store.s.grad = torch.full_like(store.s, -1.0)  # push outward
        store.t.grad = torch.full_like(store.t, -1.0)
        optimizer.step()
    assert store.s.max().item() <= 1.0 + 1e-12  # the declared domain box
    assert store.t.max().item() <= 1.0 + 1e-12


def test_repulsion_inside_the_optimizer_separates_overlapping_twins() -> None:
    # exactly coincident twins are an unstable equilibrium (the pair force
    # vanishes with the displacement), so the twins start a hair apart
    module, store = _line_site([0.50, 0.52], [0.50, 0.50], [0.0, 0.0])
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=0.5,
        target_step=0.05,
        repulsion=PairRepulsion(0.5, pairs=1 << 16),
        seed=11,
    )
    for _ in range(100):
        store.s.grad = torch.zeros_like(store.s)
        store.t.grad = torch.zeros_like(store.t)
        optimizer.step()
    joint = (
        (store.s[0] - store.s[1]).square().sum()
        + (store.t[0] - store.t[1]).square().sum()
    ).sqrt()
    assert joint.item() > SIGMA  # zero-loss twins were pushed apart


def test_decoupled_repulsion_moves_without_entering_adam_moments() -> None:
    module, store = _line_site([0.50, 0.52], [0.50, 0.50], [0.0, 0.0])
    before = (store.s[0] - store.s[1]).abs().item()
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=0.5,
        target_step=0.05,
        decoupled_repulsion=PairRepulsion(0.01, pairs=1 << 16),
        seed=11,
    )
    store.s.grad = torch.zeros_like(store.s)
    store.t.grad = torch.zeros_like(store.t)
    optimizer.step()
    assert optimizer.m_s is not None and optimizer.v_s is not None
    assert torch.count_nonzero(optimizer.m_s) == 0
    assert torch.count_nonzero(optimizer.v_s) == 0
    assert (store.s[0] - store.s[1]).abs().item() > before


def test_structural_state_and_follower_contract() -> None:
    module, store = _line_site([0.2, 0.8], [0.2, 0.8], [0.5, 0.4])
    optimizer = PullbackAdam(
        module,
        moment_space="tangent",
        cap_sigma=0.1,
        target_step=0.02,
        rent=SmoothRent(0.05),
        lr_w=0.03,
    )
    store.s.grad = torch.full_like(store.s, 0.1)
    store.t.grad = torch.full_like(store.t, -0.1)
    store.w.grad = torch.full_like(store.w, 0.2)
    optimizer.step()
    assert optimizer.m_w is not None
    assert torch.count_nonzero(optimizer.m_w) > 0

    state = optimizer.state_dict()
    assert state["owns_w"] is True
    twin_module, _ = _line_site([0.2, 0.8], [0.2, 0.8], [0.5, 0.4])
    restored = PullbackAdam(
        twin_module,
        moment_space="tangent",
        cap_sigma=0.1,
        target_step=0.02,
        rent=SmoothRent(0.05),
        lr_w=0.03,
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.m_w, optimizer.m_w)

    plain_module, _ = _line_site([0.2, 0.8], [0.2, 0.8], [0.5, 0.4])
    plain = PullbackAdam(
        plain_module, moment_space="tangent", cap_sigma=0.1
    )
    with pytest.raises(ValueError, match="rent ownership"):
        plain.load_state_dict(state)

    optimizer.on_refit(torch.tensor([0]))
    assert torch.count_nonzero(optimizer.m_w[0]) == 0
    assert torch.count_nonzero(optimizer.m_s) > 0  # coordinates untouched
    optimizer.on_death(torch.tensor([1]))
    for name in ("m_s", "m_t", "v_s", "v_t", "m_w", "v_w", "travel"):
        assert torch.count_nonzero(getattr(optimizer, name)[1]) == 0

    optimizer.zero_grad()
    assert store.w.grad is None  # rent optimizer owns the amplitude grads


def test_decoupled_decay_owns_amplitudes_without_a_rent() -> None:
    """A price outside the moments is a separate choice from a price inside.

    Which optimizer owns ``w`` and where the price is charged were one
    decision while ``rent`` alone conferred ownership; ``decay`` is the
    other half, and either one is enough to take the amplitudes.
    """
    module, store = _line_site([0.2], [0.2], [0.5])
    optimizer = PullbackAdam(
        module, moment_space="tangent", cap_sigma=0.1,
        decay=0.5, lr_w=0.01,
    )

    assert optimizer.owns_amplitudes
    assert optimizer.rent is None

    before = store.w.detach().clone()
    store.s.grad = torch.full_like(store.s, 0.1)
    store.t.grad = torch.full_like(store.t, -0.1)
    store.w.grad = torch.full_like(store.w, 0.2)
    optimizer.step()

    assert not torch.equal(store.w.detach(), before)
    assert optimizer.m_w is not None  # amplitude moments were allocated
    optimizer.zero_grad()
    assert store.w.grad is None


def test_decoupled_decay_shrinks_by_its_own_rate() -> None:
    """``w -= lr_w * decay * w``, charged after the Adam step.

    With no loss gradient on the amplitudes the Adam term contributes
    nothing, so what remains is exactly the decoupled factor -- the point of
    decoupling being that ``sqrt(v)`` never touches it.
    """
    module, store = _line_site([0.2], [0.2], [0.5])
    optimizer = PullbackAdam(
        module, moment_space="tangent", cap_sigma=0.1,
        decay=0.5, lr_w=0.01,
    )
    before = store.w.detach().clone()
    store.s.grad = torch.zeros_like(store.s)
    store.t.grad = torch.zeros_like(store.t)
    store.w.grad = torch.zeros_like(store.w)  # only the price acts
    optimizer.step()

    torch.testing.assert_close(
        store.w.detach(), before * (1.0 - 0.01 * 0.5), rtol=1e-6, atol=0.0
    )


def test_rejects_inconsistent_structural_configuration() -> None:
    module, _ = _line_site([0.2], [0.2], [0.5])
    with pytest.raises(ValueError, match="lr_w comes with"):
        PullbackAdam(
            module,
            moment_space="tangent",
            cap_sigma=0.1,
            rent=SmoothRent(0.1),
        )
    with pytest.raises(ValueError, match="lr_w comes with"):
        PullbackAdam(
            module, moment_space="tangent", cap_sigma=0.1, lr_w=0.01
        )
    with pytest.raises(ValueError, match="lr_w comes with"):
        PullbackAdam(
            module, moment_space="tangent", cap_sigma=0.1, decay=0.5
        )
    with pytest.raises(ValueError, match="decay"):
        PullbackAdam(
            module, moment_space="tangent", cap_sigma=0.1,
            decay=-1.0, lr_w=0.01,
        )
    with pytest.raises(TypeError, match="repulsion"):
        PullbackAdam(
            module,
            moment_space="tangent",
            cap_sigma=0.1,
            repulsion=object(),
        )
    with pytest.raises(TypeError, match="wall"):
        PullbackAdam(
            module, moment_space="tangent", cap_sigma=0.1, wall=1
        )
