"""ChartPullbackAdam: the chart metric as a sum of incident pullbacks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchcst import ChartPullbackAdam, ChartRepulsion
from torchcst.compute import CSTLinear, Materialized
from torchcst.representation import (
    GaussianKernel,
    L2NormalizedColumns,
    RepresentationSpec,
)
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore

SIGMA = 0.35


def _birth(store, source, target, weights, lineage_start=0):
    count = len(weights)
    return SynapseBirth(
        store.site,
        torch.as_tensor(source, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
        torch.as_tensor(weights, dtype=torch.float64),
        torch.arange(lineage_start, lineage_start + count, dtype=torch.int64),
    )


def _synapses(name, atoms, d, generator, *, source=None, target=None):
    store = SynapseStore(
        name,
        d,
        d,
        atoms,
        spec=RepresentationSpec.continuous(d, d),
        dtype=torch.float64,
    )
    if source is None:
        source = torch.rand(atoms, d, generator=generator, dtype=torch.float64)
    if target is None:
        target = torch.rand(atoms, d, generator=generator, dtype=torch.float64)
    weights = 0.5 + torch.rand(atoms, generator=generator, dtype=torch.float64)
    store.apply([_birth(store, source, target, weights)])
    return store


def _chart(name, count, d, generator, *, live=None):
    mu = nn.Parameter(
        torch.rand(count, d, generator=generator, dtype=torch.float64)
    )
    return NeuronStore(
        name, count, mu=mu,
        initial_live=count if live is None else live,
        dtype=torch.float64,
    )


def _pair(*, atoms=1, d=2, seed=11, live_hid=None, sigma=SIGMA):
    """``up: res -> hid`` and ``down: hid -> res``, sharing both charts."""
    generator = torch.Generator().manual_seed(seed)
    res = _chart("chart-res", 5, d, generator)
    hid = _chart("chart-hid", 7, d, generator, live=live_hid)
    up = CSTLinear(
        res, hid, _synapses("chart-up", atoms, d, generator),
        GaussianKernel(sigma).double(),
        gauge=L2NormalizedColumns(),
        backend=Materialized(lean=False),
    )
    down = CSTLinear(
        hid, res, _synapses("chart-down", atoms, d, generator),
        GaussianKernel(sigma).double(),
        gauge=L2NormalizedColumns(),
        backend=Materialized(lean=False),
    )
    return res, hid, up, down


def _weight_fn(site, chart):
    """Delivered weight as a pure function of the chart coordinates."""

    def function(mu):
        original = chart._parameters["mu"]
        chart._parameters["mu"] = mu
        try:
            return site.dense_weight()
        finally:
            chart._parameters["mu"] = original

    return function


def _oracle_blocks(chart, sites):
    """Exact per-neuron ``J^T J`` summed over incident sites, via autograd.

    The whole delivered weight participates: through the column
    normalization, one chart point perturbs every row of its side's columns
    (``du = g (e_i - u u_i)``), not just its own row.
    """
    count, d = chart.mu.shape
    total = torch.zeros(count, d, d, dtype=torch.float64)
    for site in sites:
        jacobian = torch.autograd.functional.jacobian(
            _weight_fn(site, chart), chart.mu.detach().clone(), vectorize=True
        )
        # jacobian: [n_out, n_in, count, d]
        for i in range(count):
            rows = jacobian[:, :, i, :].reshape(-1, d)
            total[i] += rows.T @ rows
    return total


def test_metric_matches_autograd_on_a_single_atom():
    _, hid, up, down = _pair(atoms=1)
    optimizer = ChartPullbackAdam(hid, [up, down], metric="block")
    live = hid.live_ids()
    closed = optimizer._metric_blocks(live)
    oracle = _oracle_blocks(hid, [up, down])
    torch.testing.assert_close(closed, oracle, rtol=1e-8, atol=1e-10)


def test_metric_matches_autograd_when_atoms_are_separated():
    generator = torch.Generator().manual_seed(3)
    res = _chart("sep-res", 5, 2, generator)
    hid = _chart("sep-hid", 7, 2, generator)
    corners = torch.tensor(
        [[0.15, 0.15], [0.85, 0.85]], dtype=torch.float64
    )
    up = CSTLinear(
        res, hid,
        _synapses("sep-up", 2, 2, generator, source=corners, target=corners),
        GaussianKernel(0.05).double(),
        gauge=L2NormalizedColumns(),
        backend=Materialized(lean=False),
    )
    optimizer = ChartPullbackAdam(hid, [up], metric="block")
    closed = optimizer._metric_blocks(hid.live_ids())
    oracle = _oracle_blocks(hid, [up])
    torch.testing.assert_close(closed, oracle, rtol=1e-4, atol=1e-9)


def test_metric_sums_the_incident_sites():
    _, hid, up, down = _pair(atoms=3)
    live = hid.live_ids()
    both = ChartPullbackAdam(hid, [up, down])._metric_blocks(live)
    only_up = ChartPullbackAdam(hid, [up])._metric_blocks(live)
    only_down = ChartPullbackAdam(hid, [down])._metric_blocks(live)
    torch.testing.assert_close(both, only_up + only_down)


def test_gradient_arrives_summed_over_incidences():
    _, hid, up, down = _pair(atoms=3)
    up.dense_weight().sum().backward()
    only_up = hid.mu.grad.clone()
    hid.mu.grad = None
    down.dense_weight().sum().backward()
    only_down = hid.mu.grad.clone()
    hid.mu.grad = None
    (up.dense_weight().sum() + down.dense_weight().sum()).backward()
    torch.testing.assert_close(hid.mu.grad, only_up + only_down)


def test_first_step_median_displacement_is_target():
    _, hid, up, down = _pair(atoms=4)
    optimizer = ChartPullbackAdam(hid, [up, down], target_step=0.01)
    generator = torch.Generator().manual_seed(5)
    hid.mu.grad = torch.randn(
        hid.mu.shape, generator=generator, dtype=torch.float64
    )
    before = hid.mu.detach().clone()
    optimizer.step()
    moved = (hid.mu.detach() - before).norm(dim=1)
    torch.testing.assert_close(
        moved.median(), torch.tensor(0.01 * SIGMA, dtype=torch.float64)
    )


def test_cap_bounds_every_neuron():
    _, hid, up, down = _pair(atoms=4)
    optimizer = ChartPullbackAdam(
        hid, [up, down], target_step=50.0, cap_sigma=0.1
    )
    generator = torch.Generator().manual_seed(6)
    hid.mu.grad = torch.randn(
        hid.mu.shape, generator=generator, dtype=torch.float64
    )
    before = hid.mu.detach().clone()
    optimizer.step()
    moved = (hid.mu.detach() - before).norm(dim=1)
    assert bool((moved <= 0.1 * SIGMA + 1e-12).all())


def test_repulsion_matches_autograd_and_pushes_apart():
    force = ChartRepulsion(0.5)
    coords = torch.tensor(
        [[0.50, 0.50], [0.52, 0.50], [0.90, 0.90]], dtype=torch.float64
    )
    live = torch.arange(3)
    sigma = torch.tensor(SIGMA, dtype=torch.float64)

    tracked = coords.clone().requires_grad_(True)
    diff = (tracked[:, None, :] - tracked[None, :, :]) / sigma
    energy = 0.5 * torch.exp(-0.5 * diff.square().sum(-1))
    potential = torch.triu(energy, diagonal=1).sum()
    potential.backward()
    torch.testing.assert_close(
        force.gradient(coords, live, sigma, None), tracked.grad
    )

    _, hid, up, down = _pair(atoms=2, live_hid=2)
    with torch.no_grad():
        hid.mu[0] = torch.tensor([0.50, 0.50], dtype=torch.float64)
        hid.mu[1] = torch.tensor([0.51, 0.50], dtype=torch.float64)
    optimizer = ChartPullbackAdam(
        hid, [up, down], repulsion=ChartRepulsion(1.0)
    )
    gap_before = (hid.mu[0] - hid.mu[1]).norm().item()
    hid.mu.grad = torch.zeros_like(hid.mu)
    optimizer.step()
    gap_after = (hid.mu[0] - hid.mu[1]).norm().item()
    assert gap_after > gap_before


def test_wall_clamps_into_the_chart_box():
    _, hid, up, down = _pair(atoms=2)
    with torch.no_grad():
        hid.mu[0] = torch.tensor([2.0, -1.0], dtype=torch.float64)
    optimizer = ChartPullbackAdam(hid, [up, down], wall=True)
    hid.mu.grad = torch.zeros_like(hid.mu)
    optimizer.step()
    lo = hid.mu.new_tensor(optimizer.box.lo_per_axis)
    hi = hid.mu.new_tensor(optimizer.box.hi_per_axis)
    assert bool((hid.mu.detach() >= lo - 1e-12).all())
    assert bool((hid.mu.detach() <= hi + 1e-12).all())


def test_dormant_neurons_do_not_move():
    _, hid, up, down = _pair(atoms=3, live_hid=4)
    optimizer = ChartPullbackAdam(hid, [up, down])
    before = hid.mu.detach().clone()
    hid.mu.grad = torch.ones_like(hid.mu)
    optimizer.step()
    live = set(hid.live_ids().tolist())
    for row in range(hid.mu.shape[0]):
        if row not in live:
            torch.testing.assert_close(hid.mu.detach()[row], before[row])
            assert optimizer.travel[row].item() == 0.0


def test_state_dict_roundtrip_and_moment_space_guard():
    _, hid, up, down = _pair(atoms=2)
    optimizer = ChartPullbackAdam(hid, [up, down])
    hid.mu.grad = torch.ones_like(hid.mu)
    optimizer.step()
    state = optimizer.state_dict()

    fresh = ChartPullbackAdam(hid, [up, down])
    fresh.load_state_dict(state)
    assert fresh.step_count == optimizer.step_count
    assert fresh.eta == optimizer.eta
    torch.testing.assert_close(fresh.m, optimizer.m)

    other = ChartPullbackAdam(hid, [up, down], moment_space="parameter")
    with pytest.raises(ValueError):
        other.load_state_dict(state)


def test_rejects_frozen_charts_wrong_gauge_and_strangers():
    generator = torch.Generator().manual_seed(9)
    frozen = NeuronStore(
        "frozen", 4,
        mu=torch.rand(4, 2, generator=generator, dtype=torch.float64),
        initial_live=4, dtype=torch.float64,
    )
    res, hid, up, _down = _pair(atoms=2)
    with pytest.raises(TypeError):
        ChartPullbackAdam(frozen, [up])
    with pytest.raises(ValueError):
        ChartPullbackAdam(hid, [])
    _res, _hid, other_up, _ = _pair(atoms=2, seed=23)
    with pytest.raises(ValueError):
        ChartPullbackAdam(hid, [other_up])
    bare = CSTLinear(
        res, hid, _synapses("bare", 2, 2, generator),
        GaussianKernel(SIGMA).double(),
        backend=Materialized(lean=False),
    )
    with pytest.raises(TypeError):
        ChartPullbackAdam(hid, [bare])
