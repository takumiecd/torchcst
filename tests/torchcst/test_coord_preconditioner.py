"""CoordPreconditioner: frozen lm1d update rule + follower contract.

The update rule is the lm1d-frozen Gaussian-metric preconditioner (confirmed
3/3 seeds in lm1e).  The golden test below recomputes it with an independent
naive reference -- including the [N, K, d] distance cube the implementation
deliberately avoids -- so a silent drift in the closed form breaks loudly.
"""

from __future__ import annotations

import pytest
import torch

from torchcst import CoordPreconditioner
from torchcst.compute import CSTLinear
from torchcst.representation import (
    GaussianFactor,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore

SIGMA = 0.25


def _birth(store, source, target, weights, lineage_start=0):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def _site(capacity=4, atoms=3, d_in=2, d_out=1, gauge=None):
    """A double-precision continuous site with ``atoms`` live rows."""
    store = SynapseStore(
        "precond",
        d_in,
        d_out,
        capacity,
        spec=RepresentationSpec.continuous(d_in, d_out),
        dtype=torch.float64,
    )
    rng = torch.Generator().manual_seed(7)
    source = torch.rand(atoms, d_in, generator=rng, dtype=torch.float64)
    target = torch.rand(atoms, d_out, generator=rng, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=rng, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    inputs = NeuronStore(
        "inputs",
        5,
        mu=torch.rand(5, d_in, generator=rng, dtype=torch.float64),
        initial_live=5,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "outputs",
        4,
        mu=torch.rand(4, d_out, generator=rng, dtype=torch.float64),
        initial_live=4,
        dtype=torch.float64,
    )
    module = CSTLinear(
        inputs, outputs, store, GaussianFactor(SIGMA).double(), gauge=gauge
    )
    return module, store


def _backward(module, seed):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(2, module.in_features, generator=rng, dtype=torch.float64)
    module(x).sum().backward()


def _reference_jacobian_sq(module, store):
    """The lm1d J^2, via the explicit [N, K, d] cube the closed form avoids."""
    sigma = float(module.factor_in.sigma.detach())
    mu_in = module.in_neurons.mu
    mu_out = module.out_neurons.mu
    k_in = module.factor_in(mu_in, store.s.detach())
    k_out = module.factor_out(mu_out, store.t.detach())
    d_in_cube = (mu_in[:, None, :] - store.s.detach()[None, :, :]).square().sum(-1)
    d_out_cube = (mu_out[:, None, :] - store.t.detach()[None, :, :]).square().sum(-1)
    r_in = (k_in.square() * d_in_cube).sum(0)
    r_out = (k_out.square() * d_out_cube).sum(0)
    w_sq = store.w.detach().square()
    j_s = w_sq * k_out.square().sum(0) * r_in / sigma**4
    j_t = w_sq * k_in.square().sum(0) * r_out / sigma**4
    return j_s, j_t


def _reference_step(module, store, v_s, v_t, eta, live, *, beta, target_step,
                    cap_sigma, lr_scale=1.0):
    """One frozen-rule update, returning (delta_s, delta_t, v_s, v_t, eta)."""
    sigma = float(module.factor_in.sigma.detach())
    v_s = beta * v_s + (1.0 - beta) * store.s.grad
    v_t = beta * v_t + (1.0 - beta) * store.t.grad
    j_s, j_t = _reference_jacobian_sq(module, store)
    lam_s = j_s[live].median().clamp(min=1e-30)
    lam_t = j_t[live].median().clamp(min=1e-30)
    raw_s = v_s / (j_s + lam_s)[:, None]
    raw_t = v_t / (j_t + lam_t)[:, None]
    if eta is None:
        med = torch.cat(
            [raw_s[live].norm(dim=1), raw_t[live].norm(dim=1)]
        ).median()
        eta = target_step * sigma / float(med)
    deltas = []
    for raw in (raw_s, raw_t):
        delta = raw * (eta * lr_scale)
        norms = delta.norm(dim=1, keepdim=True).clamp(min=1e-30)
        delta = delta * (norms.clamp(max=cap_sigma * sigma) / norms)
        masked = torch.zeros_like(delta)
        masked[live] = delta[live]
        deltas.append(masked)
    return deltas[0], deltas[1], v_s, v_t, eta


def test_two_steps_match_the_naive_reference() -> None:
    module, store = _site()
    pc = CoordPreconditioner(module, cap_sigma=0.1)
    live = store.live_slots()

    v_s = torch.zeros_like(store.s)
    v_t = torch.zeros_like(store.t)
    eta = None
    for seed, lr_scale in ((11, 1.0), (12, 0.4)):
        _backward(module, seed)
        s_before = store.s.detach().clone()
        t_before = store.t.detach().clone()
        exp_ds, exp_dt, v_s, v_t, eta = _reference_step(
            module, store, v_s, v_t, eta, live,
            beta=0.9, target_step=0.01, cap_sigma=0.1, lr_scale=lr_scale,
        )
        pc.step(lr_scale)
        pc.zero_grad()
        torch.testing.assert_close(store.s.detach(), s_before - exp_ds)
        torch.testing.assert_close(store.t.detach(), t_before - exp_dt)
    assert pc.eta == pytest.approx(eta)
    torch.testing.assert_close(pc.v_s, v_s)
    torch.testing.assert_close(pc.v_t, v_t)


def test_preconditioner_uses_the_normalized_map_jacobian_under_the_l2_gauge():
    module, store = _site(gauge=L2NormalizedColumns())
    precond = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    actual_s, actual_t = precond._jacobian_sq()
    gauge = L2NormalizedColumns()

    expected_s = []
    expected_t = []
    for slot in range(store.s.shape[0]):
        weight = store.w.detach()[slot]
        source = store.s.detach()[slot]
        target = store.t.detach()[slot]

        def represented(s, t, weight=weight):
            column_in = gauge.columns(
                module.factor_in, module.in_neurons.mu, s[None, :]
            )[:, 0]
            column_out = gauge.columns(
                module.factor_out, module.out_neurons.mu, t[None, :]
            )[:, 0]
            return weight * column_out[:, None] * column_in[None, :]

        jac_s = torch.autograd.functional.jacobian(
            lambda s, target=target: represented(s, target), source
        )
        jac_t = torch.autograd.functional.jacobian(
            lambda t, source=source: represented(source, t), target
        )
        expected_s.append(jac_s.square().sum())
        expected_t.append(jac_t.square().sum())

    torch.testing.assert_close(actual_s, torch.stack(expected_s))
    torch.testing.assert_close(actual_t, torch.stack(expected_t))


def test_eta_calibration_puts_the_live_median_step_at_target() -> None:
    module, store = _site()
    pc = CoordPreconditioner(module, cap_sigma=10.0, target_step=0.01)
    _backward(module, 21)
    s_before = store.s.detach().clone()
    t_before = store.t.detach().clone()
    pc.step(1.0)
    live = store.live_slots()
    steps = torch.cat(
        [
            (store.s.detach() - s_before)[live].norm(dim=1),
            (store.t.detach() - t_before)[live].norm(dim=1),
        ]
    )
    sigma = float(module.factor_in.sigma.detach())
    assert float(steps.median()) == pytest.approx(0.01 * sigma, rel=1e-6)


def test_trust_cap_bounds_every_atom_step() -> None:
    module, store = _site()
    pc = CoordPreconditioner(module, cap_sigma=0.02, target_step=5.0)
    for seed in (31, 32, 33):
        _backward(module, seed)
        s_before = store.s.detach().clone()
        t_before = store.t.detach().clone()
        pc.step(1.0)
        pc.zero_grad()
        cap = 0.02 * float(module.factor_in.sigma.detach())
        for before, param in ((s_before, store.s), (t_before, store.t)):
            norms = (param.detach() - before).norm(dim=1)
            assert bool((norms <= cap * (1.0 + 1e-12)).all())


def test_dead_slots_neither_move_nor_skew_the_damping() -> None:
    module, store = _site(capacity=4, atoms=4)
    dead_id = store.live_ids()[1:2]
    store.apply([SynapseDeath(store.site, dead_id)])
    dead_slot = 1
    live = store.live_slots()
    assert dead_slot not in live.tolist()

    # Poison the dead slot: stale coordinates, weight, and gradient must all
    # be invisible to the statistics and to the update.
    with torch.no_grad():
        store.w[dead_slot] = 50.0
    _backward(module, 41)
    store.s.grad[dead_slot] = 3.0
    store.t.grad[dead_slot] = 3.0

    # Twin site holding only the three surviving atoms, same everything.
    twin_module, twin_store = _site(capacity=4, atoms=4)
    twin_store.apply([SynapseDeath(twin_store.site, dead_id)])
    _backward(twin_module, 41)

    pc = CoordPreconditioner(module, cap_sigma=0.1)
    twin = CoordPreconditioner(twin_module, cap_sigma=0.1)
    s_before = store.s.detach().clone()
    pc.step(1.0)
    twin.step(1.0)

    torch.testing.assert_close(store.s.detach()[dead_slot], s_before[dead_slot])
    torch.testing.assert_close(
        store.s.detach()[live], twin_store.s.detach()[live]
    )
    torch.testing.assert_close(
        store.t.detach()[live], twin_store.t.detach()[live]
    )
    assert float(pc.travel[dead_slot]) == 0.0


def test_follower_zeroes_momentum_on_slot_reuse_and_pads_on_growth() -> None:
    module, store = _site(capacity=4, atoms=3)
    pc = CoordPreconditioner(module, cap_sigma=0.1)
    for seed in (51, 52):
        _backward(module, seed)
        pc.step(1.0)
        pc.zero_grad()
    survivor_slot = 2
    preserved = (
        pc.v_s[survivor_slot].clone(),
        pc.v_t[survivor_slot].clone(),
        pc.travel[survivor_slot].clone(),
    )

    # Kill the first atom and rebirth into the freed slot in one event.
    victim = store.live_ids()[:1]
    store.apply(
        [
            SynapseDeath(store.site, victim),
            _birth(store, [[0.5, 0.5]], [[0.5]], [1.0], lineage_start=10),
        ]
    )
    reused_slot = 0
    assert bool(torch.count_nonzero(pc.v_s[reused_slot]) == 0)
    assert bool(torch.count_nonzero(pc.v_t[reused_slot]) == 0)
    assert float(pc.travel[reused_slot]) == 0.0
    torch.testing.assert_close(pc.v_s[survivor_slot], preserved[0])
    torch.testing.assert_close(pc.v_t[survivor_slot], preserved[1])
    torch.testing.assert_close(pc.travel[survivor_slot], preserved[2])

    # Births beyond capacity grow the store; state rows must follow.
    extra = torch.full((4, 2), 0.25, dtype=torch.float64)
    store.apply(
        [_birth(store, extra, torch.full((4, 1), 0.75), [1.0] * 4,
                lineage_start=20)]
    )
    assert store.capacity > 4
    assert pc.v_s.shape[0] == store.capacity
    assert pc.v_t.shape[0] == store.capacity
    assert pc.travel.shape[0] == store.capacity
    torch.testing.assert_close(pc.v_s[survivor_slot], preserved[0])

    # The grown state keeps stepping without shape errors.
    _backward(module, 53)
    pc.step(1.0)


def test_rejected_mutation_leaves_state_untouched() -> None:
    module, store = _site(capacity=4, atoms=3)
    pc = CoordPreconditioner(module, cap_sigma=0.1)
    _backward(module, 61)
    pc.step(1.0)
    snapshot = (pc.v_s.clone(), pc.v_t.clone(), pc.travel.clone())

    bogus = torch.tensor([987654], dtype=torch.int64)
    with pytest.raises(Exception):
        store.apply([SynapseDeath(store.site, bogus)])

    torch.testing.assert_close(pc.v_s, snapshot[0])
    torch.testing.assert_close(pc.v_t, snapshot[1])
    torch.testing.assert_close(pc.travel, snapshot[2])


def test_state_dict_roundtrip_restores_the_trajectory() -> None:
    module, store = _site()
    pc = CoordPreconditioner(module, cap_sigma=0.1)
    _backward(module, 71)
    pc.step(1.0)
    pc.zero_grad()
    state = pc.state_dict()
    coords = (store.s.detach().clone(), store.t.detach().clone())

    # A fresh instance on the same site, restored, must continue identically.
    resumed = CoordPreconditioner(module, cap_sigma=0.1, subscribe=False)
    resumed.load_state_dict(state)
    _backward(module, 72)
    grads = (store.s.grad.clone(), store.t.grad.clone())
    resumed.step(1.0)
    resumed_coords = (store.s.detach().clone(), store.t.detach().clone())

    # Rewind and replay the same gradient through the original instance.
    with torch.no_grad():
        store.s.copy_(coords[0])
        store.t.copy_(coords[1])
    store.s.grad, store.t.grad = grads
    pc.step(1.0)
    torch.testing.assert_close(store.s.detach(), resumed_coords[0])
    torch.testing.assert_close(store.t.detach(), resumed_coords[1])


def test_division_of_labor_with_an_optimizer() -> None:
    """Coordinates move only through the preconditioner, w only through Adam."""
    module, store = _site()
    pc = CoordPreconditioner(module, cap_sigma=0.1)
    optimizer = torch.optim.Adam([store.w], lr=1e-2)
    _backward(module, 81)
    s_before = store.s.detach().clone()
    w_before = store.w.detach().clone()

    optimizer.step()
    assert torch.equal(store.s.detach(), s_before)
    assert not torch.equal(store.w.detach(), w_before)

    pc.step(1.0)
    assert not torch.equal(store.s.detach(), s_before)

    optimizer.zero_grad(set_to_none=True)
    pc.zero_grad()
    assert store.s.grad is None and store.t.grad is None


def test_rejects_non_continuous_sites_and_bad_dials() -> None:
    module, _ = _site()
    with pytest.raises(TypeError):
        CoordPreconditioner(object(), cap_sigma=0.1)
    with pytest.raises(ValueError):
        CoordPreconditioner(module, cap_sigma=0.0)
    with pytest.raises(ValueError):
        CoordPreconditioner(module, cap_sigma=0.1, beta=1.0)
    with pytest.raises(ValueError):
        CoordPreconditioner(module, cap_sigma=0.1, target_step=-1.0)
