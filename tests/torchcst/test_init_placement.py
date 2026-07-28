"""Stage 4: coverage_lattice / logdet_greedy / variance_amplitudes."""

from __future__ import annotations

import math

import torch

from torchcst.compute import CSTLinear
from torchcst.init import (
    coverage_lattice,
    logdet_greedy,
    to_synapse_births,
    variance_amplitudes,
)
from torchcst.representation import GaussianKernel, GramService, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseStore


# ---------------------------------------------------------------------------
# coverage_lattice
# ---------------------------------------------------------------------------


def test_coverage_lattice_spacing_law_quadruples_count_when_sigma_halves() -> None:
    s1, t1 = coverage_lattice((0.0, 1.0), (0.0, 1.0), 1, 1, 0.1, 0.1, spacing=1.0)
    s2, t2 = coverage_lattice((0.0, 1.0), (0.0, 1.0), 1, 1, 0.05, 0.05, spacing=1.0)
    count1 = s1.shape[0]
    count2 = s2.shape[0]
    ratio = count2 / count1
    assert 2.8 <= ratio <= 5.2, f"expected ~4x growth, got {ratio} ({count1} -> {count2})"


def test_coverage_lattice_is_deterministic() -> None:
    args = ((0.0, 1.0), (-1.0, 2.0), 1, 2, 0.2, 0.3)
    s1, t1 = coverage_lattice(*args, spacing=0.8)
    s2, t2 = coverage_lattice(*args, spacing=0.8)
    assert torch.equal(s1, s2)
    assert torch.equal(t1, t2)

    s3, t3 = coverage_lattice(*args, K=40)
    s4, t4 = coverage_lattice(*args, K=40)
    assert torch.equal(s3, s4)
    assert torch.equal(t3, t4)


def test_coverage_lattice_atoms_are_inside_bounds() -> None:
    bounds_in = (0.0, 2.0)
    bounds_out = (-1.0, 1.0)
    for kwargs in ({"spacing": 0.5}, {"K": 30}):
        s, t = coverage_lattice(bounds_in, bounds_out, 2, 1, 0.3, 0.2, **kwargs)
        assert bool((s >= bounds_in[0]).all()) and bool((s <= bounds_in[1]).all())
        assert bool((t >= bounds_out[0]).all()) and bool((t <= bounds_out[1]).all())


def test_coverage_lattice_rejects_ambiguous_or_missing_stop_condition() -> None:
    try:
        coverage_lattice((0.0, 1.0), (0.0, 1.0), 1, 1, 0.1, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neither K nor spacing is given")
    try:
        coverage_lattice((0.0, 1.0), (0.0, 1.0), 1, 1, 0.1, 0.1, K=5, spacing=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both K and spacing are given")


# ---------------------------------------------------------------------------
# logdet_greedy: dense brute-force cross-check
# ---------------------------------------------------------------------------


def _dense_identity_gram(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return (U.transpose(0, 1) @ U) * (V.transpose(0, 1) @ V)


def _brute_force_greedy(
    gram_ridged: torch.Tensor, k: int
) -> tuple[list[int], list[float]]:
    """Reference greedy selection via dense ``slogdet`` differences."""
    m = gram_ridged.shape[0]
    remaining = list(range(m))
    selected: list[int] = []
    gains: list[float] = []
    prev_logdet = 0.0  # logdet of the empty (0x0) selected block
    for _ in range(k):
        best = None
        for z in remaining:
            idx = torch.tensor(selected + [z], dtype=torch.int64)
            sub = gram_ridged.index_select(0, idx).index_select(1, idx)
            _, logdet = torch.linalg.slogdet(sub)
            gain = float(logdet) - prev_logdet
            if best is None or gain > best[0]:
                best = (gain, z, float(logdet))
        gain, z, logdet = best
        selected.append(z)
        gains.append(gain)
        remaining.remove(z)
        prev_logdet = logdet
    return selected, gains


def test_logdet_greedy_matches_dense_brute_force() -> None:
    torch.manual_seed(0)
    m, n_out, n_in = 8, 3, 3
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.randn(n_in, m, dtype=torch.float64)
    s = torch.randn(m, 1, dtype=torch.float64)
    t = torch.randn(m, 1, dtype=torch.float64)
    ridge = 1e-3
    k = 5

    selection = logdet_greedy(s, t, U=U, V=V, K=k, ridge=ridge, jitter=1e-12)

    gram_ridged = _dense_identity_gram(U, V) + ridge * torch.eye(m, dtype=torch.float64)
    bf_indices, bf_gains = _brute_force_greedy(gram_ridged, k)

    assert selection.indices.tolist() == bf_indices
    torch.testing.assert_close(
        selection.gains, torch.tensor(bf_gains, dtype=torch.float64), atol=1e-8, rtol=0
    )


def test_logdet_greedy_never_buys_a_duplicate() -> None:
    torch.manual_seed(1)
    m, n_out, n_in = 6, 3, 3
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.randn(n_in, m, dtype=torch.float64)
    # index 3 is an exact duplicate of index 0.
    U[:, 3] = U[:, 0]
    V[:, 3] = V[:, 0]
    s = torch.randn(m, 1, dtype=torch.float64)
    t = torch.randn(m, 1, dtype=torch.float64)
    s[3] = s[0]
    t[3] = t[0]

    # jitter tuned above the duplicate's post-collapse residual (~ridge) so
    # the eps-null guard actually excludes it rather than merely deferring it.
    selection = logdet_greedy(s, t, U=U, V=V, K=m, ridge=1e-6, jitter=1e-4)
    indices = selection.indices.tolist()
    assert 3 not in indices
    assert 0 in indices

    # With a looser jitter the duplicate is still never bought before its twin.
    loose = logdet_greedy(s, t, U=U, V=V, K=m, ridge=1e-6, jitter=1e-8)
    loose_indices = loose.indices.tolist()
    if 3 in loose_indices:
        assert loose_indices.index(0) < loose_indices.index(3)
        # and its gain has collapsed relative to every non-duplicate selection
        dup_gain = loose.gains[loose_indices.index(3)]
        assert dup_gain < min(
            g for idx, g in zip(loose_indices, loose.gains.tolist()) if idx not in (0, 3)
        )


def test_logdet_greedy_rent_stop_matches_gain_crossing() -> None:
    torch.manual_seed(2)
    m, n_out, n_in = 10, 4, 4
    U = torch.randn(n_out, m, dtype=torch.float64) * 0.1 + torch.eye(n_out, m, dtype=torch.float64)
    V = torch.randn(n_in, m, dtype=torch.float64) * 0.1 + torch.eye(n_in, m, dtype=torch.float64)
    s = torch.arange(m, dtype=torch.float64).reshape(-1, 1)
    t = torch.arange(m, dtype=torch.float64).reshape(-1, 1)
    ridge = 1e-3

    full = logdet_greedy(s, t, U=U, V=V, K=m, ridge=ridge, jitter=1e-10)

    # Diminishing-returns SET function guarantees non-increasing per-step gain
    # is NOT true in general; it is true here by construction (a well-
    # separated, near-orthogonal candidate pool), so we test it on this pool
    # specifically rather than asserting it as a universal property.
    gains = full.gains.tolist()
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))

    i = 3
    rent = math.exp((gains[i] + gains[i + 1]) / 2.0)
    stopped = logdet_greedy(s, t, U=U, V=V, K=m, rent=rent, ridge=ridge, jitter=1e-10)
    assert stopped.indices.tolist() == full.indices.tolist()[: i + 1]


def test_logdet_greedy_avoids_data_invisible_candidates() -> None:
    torch.manual_seed(3)
    n_in, n_out, n = 4, 3, 200
    projection = torch.randn(2, n_in, dtype=torch.float64)
    base = torch.randn(n, 2, dtype=torch.float64)
    data = base @ projection  # rank <= 2 batch: n_in - 2 dims are unseen

    sigma = (data.transpose(0, 1) @ data) / n
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    null_direction = eigvecs[:, eigvals.argmin()]
    row_direction = eigvecs[:, eigvals.argmax()]

    m = 6
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.zeros(n_in, m, dtype=torch.float64)
    for i in range(3):
        V[:, i] = null_direction
    for i in range(3, 6):
        V[:, i] = row_direction
    s = torch.randn(m, 1, dtype=torch.float64)
    t = torch.randn(m, 1, dtype=torch.float64)

    selection = logdet_greedy(
        s, t, U=U, V=V, K=m, ridge=1e-6, jitter=1e-4, data=data
    )
    indices = selection.indices.tolist()
    # None of the null-space-aligned candidates (0, 1, 2) get bought at all --
    # their Gamma_zz collapses to ~0 under the data metric and the jitter
    # guard screens them out entirely.
    assert set(indices).isdisjoint({0, 1, 2})
    assert set(indices) == {3, 4, 5}


# ---------------------------------------------------------------------------
# variance_amplitudes
# ---------------------------------------------------------------------------


def test_variance_amplitudes_matches_target_second_moment() -> None:
    torch.manual_seed(4)
    k_atoms = 5
    U = torch.randn(4, k_atoms, dtype=torch.float64)
    V = torch.randn(3, k_atoms, dtype=torch.float64)
    gamma = (U.transpose(0, 1) @ U) * (V.transpose(0, 1) @ V)
    target = 2.5

    generator = torch.Generator().manual_seed(42)
    w = variance_amplitudes(U=U, V=V, target_second_moment=target, generator=generator)
    achieved = float(w @ gamma @ w)
    assert math.isclose(achieved, target, rel_tol=1e-8, abs_tol=1e-8)


def test_variance_amplitudes_is_deterministic_given_generator_state() -> None:
    torch.manual_seed(5)
    k_atoms = 6
    U = torch.randn(4, k_atoms, dtype=torch.float64)
    V = torch.randn(3, k_atoms, dtype=torch.float64)

    gen1 = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(7)
    w1 = variance_amplitudes(U=U, V=V, target_second_moment=1.0, generator=gen1)
    w2 = variance_amplitudes(U=U, V=V, target_second_moment=1.0, generator=gen2)
    assert torch.equal(w1, w2)

    gen3 = torch.Generator().manual_seed(8)
    w3 = variance_amplitudes(U=U, V=V, target_second_moment=1.0, generator=gen3)
    assert not torch.equal(w1, w3)


def test_variance_amplitudes_accepts_precomputed_gamma() -> None:
    torch.manual_seed(6)
    k_atoms = 4
    gamma = torch.eye(k_atoms, dtype=torch.float64) * 3.0
    generator = torch.Generator().manual_seed(1)
    w = variance_amplitudes(Gamma=gamma, target_second_moment=6.0, generator=generator)
    achieved = float(w @ gamma @ w)
    assert math.isclose(achieved, 6.0, rel_tol=1e-8, abs_tol=1e-8)


# ---------------------------------------------------------------------------
# Integration: coverage_lattice -> SynapseBirth -> CSTLinear -> GramService
# ---------------------------------------------------------------------------


def test_coverage_init_seeds_store_and_keeps_local_gram_conditioned() -> None:
    torch.manual_seed(7)
    d_in, d_out = 1, 1
    sigma = 0.15

    inputs = NeuronStore("inputs", 6, mu=torch.linspace(0.0, 1.0, 6)[:, None], initial_live=6)
    outputs = NeuronStore("outputs", 5, mu=torch.linspace(0.0, 1.0, 5)[:, None], initial_live=5)

    s, t = coverage_lattice((0.0, 1.0), (0.0, 1.0), d_in, d_out, sigma, sigma, spacing=1.0)
    s, t = s.float(), t.float()

    synapses = SynapseStore(
        "layer",
        d_in=d_in,
        d_out=d_out,
        capacity=s.shape[0] + 4,
        spec=RepresentationSpec.continuous(d_in, d_out),
    )
    layer = CSTLinear(inputs, outputs, synapses, GaussianKernel(sigma, learnable=False))

    k_out, k_in = layer.kernel_columns(s, t)
    generator = torch.Generator().manual_seed(0)
    w = variance_amplitudes(
        U=k_out.double(), V=k_in.double(), target_second_moment=1.0, generator=generator
    ).float()

    synapses.apply(to_synapse_births("layer", s, t, w))
    assert synapses.live_ids().numel() == s.shape[0]

    x = torch.randn(4, inputs.n_max)
    y = layer(x)
    assert y.shape == (4, outputs.n_max)

    view = synapses.view()
    u_live, v_live = layer.kernel_columns(view.s, view.t)
    gram_service = GramService(
        u_live, v_live, view.w, view.s, view.t, radius=1.5 * sigma, ridge=1e-4
    )
    lambda_mins = [gram_service.local_lambda_min(k) for k in range(gram_service.k_live)]
    assert min(lambda_mins) > 1e-4
