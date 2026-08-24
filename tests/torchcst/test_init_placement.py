"""Stage 4 of ``docs/absorb-and-gram-design.md``: init/birth-placement tests.

Covers :func:`torchcst.init.coverage_lattice`, :func:`torchcst.init.logdet_greedy`,
:func:`torchcst.init.variance_amplitudes`, and the :func:`torchcst.init.to_synapse_births`
wrapping helper, plus one end-to-end integration test that places a coverage
lattice into a real :class:`~torchcst.compute.CSTLinear` and checks the local
Gram stays well-conditioned (the design's motivating claim for Stage 4).
"""

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
from torchcst.representation import GaussianFactor, GramService, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseStore


# ---------------------------------------------------------------------------
# coverage_lattice
# ---------------------------------------------------------------------------


def test_coverage_lattice_spacing_law_halving_sigma_quadruples_count() -> None:
    # d_in = d_out = 1 makes z = (s, t) a 2-axis joint space; bounds are large
    # relative to sigma so integer-rounding noise in the per-axis point count
    # stays small, and the law shows up cleanly.
    s1, t1 = coverage_lattice((-5.0, 5.0), (-5.0, 5.0), 1, 1, 0.1, 0.1, spacing=1.0)
    s2, t2 = coverage_lattice((-5.0, 5.0), (-5.0, 5.0), 1, 1, 0.05, 0.05, spacing=1.0)
    m1, m2 = s1.shape[0], s2.shape[0]
    ratio = m2 / m1
    assert 3.0 < ratio < 5.2  # ~4x, generous tolerance for rounding


def test_coverage_lattice_is_deterministic() -> None:
    args = ((-3.0, 3.0), (-2.0, 2.0), 1, 2, 0.2, 0.3)
    s1, t1 = coverage_lattice(*args, spacing=0.7)
    s2, t2 = coverage_lattice(*args, spacing=0.7)
    assert torch.equal(s1, s2)
    assert torch.equal(t1, t2)

    s3, t3 = coverage_lattice(*args, K=200)
    s4, t4 = coverage_lattice(*args, K=200)
    assert torch.equal(s3, s4)
    assert torch.equal(t3, t4)


def test_coverage_lattice_atoms_stay_inside_bounds_both_call_paths() -> None:
    bounds_in = (-1.0, 1.0)
    bounds_out = (0.0, 2.0)
    d_in, d_out = 2, 1
    sigma_in, sigma_out = 0.15, 0.25

    s_spacing, t_spacing = coverage_lattice(
        bounds_in, bounds_out, d_in, d_out, sigma_in, sigma_out, spacing=0.9
    )
    s_k, t_k = coverage_lattice(
        bounds_in, bounds_out, d_in, d_out, sigma_in, sigma_out, K=150
    )
    for s, t in ((s_spacing, t_spacing), (s_k, t_k)):
        assert bool((s >= bounds_in[0]).all() and (s <= bounds_in[1]).all())
        assert bool((t >= bounds_out[0]).all() and (t <= bounds_out[1]).all())


def test_coverage_lattice_rejects_both_or_neither_of_k_and_spacing() -> None:
    try:
        coverage_lattice((-1.0, 1.0), (-1.0, 1.0), 1, 1, 0.1, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when neither K nor spacing given")

    try:
        coverage_lattice((-1.0, 1.0), (-1.0, 1.0), 1, 1, 0.1, 0.1, K=10, spacing=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both K and spacing given")


# ---------------------------------------------------------------------------
# logdet_greedy
# ---------------------------------------------------------------------------


def _dense_hadamard_gram(
    U: torch.Tensor, V: torch.Tensor, sigma: torch.Tensor | None
) -> torch.Tensor:
    """``Gamma_D = (U^T U) * (V^T Sigma V)`` (identity metric if sigma is None).

    This is the same Hadamard-separable formula both ``logdet_greedy`` and
    ``GramService`` use; it is an algebraic identity, not a second
    implementation of the greedy *selection logic* under test, so using it to
    build one dense ``[M, M]`` reference matrix does not make the brute-force
    check circular.
    """
    ut_u = U.transpose(0, 1) @ U
    vt_v = V.transpose(0, 1) @ (sigma @ V) if sigma is not None else V.transpose(0, 1) @ V
    return ut_u * vt_v


def test_logdet_greedy_matches_dense_brute_force_reference() -> None:
    torch.manual_seed(0)
    m, n_out, n_in = 8, 3, 3
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.randn(n_in, m, dtype=torch.float64)
    s = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    t = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    ridge = 1e-3
    k = 5

    result = logdet_greedy(s, t, U=U, V=V, K=k, ridge=ridge, jitter=1e-10)

    gamma_ridge = _dense_hadamard_gram(U, V, sigma=None) + ridge * torch.eye(
        m, dtype=torch.float64
    )
    selected: list[int] = []
    remaining = list(range(m))
    prev_logdet = 0.0  # logdet of the empty (0x0) selected block is 0
    brute_indices: list[int] = []
    brute_gains: list[float] = []
    for _ in range(k):
        best: tuple[float, int] | None = None
        for z in remaining:
            idx = torch.tensor(selected + [z], dtype=torch.int64)
            sub = gamma_ridge[idx][:, idx]
            _, logdet = torch.linalg.slogdet(sub)
            gain = float(logdet) - prev_logdet
            if best is None or gain > best[0]:
                best = (gain, z)
        gain, z = best  # type: ignore[misc]
        selected.append(z)
        remaining.remove(z)
        prev_logdet += gain
        brute_indices.append(z)
        brute_gains.append(gain)

    assert result.indices.tolist() == brute_indices
    torch.testing.assert_close(
        result.gains, torch.tensor(brute_gains, dtype=torch.float64), atol=1e-8, rtol=0.0
    )


def test_logdet_greedy_never_buys_a_duplicate() -> None:
    torch.manual_seed(3)
    m, n_out, n_in = 6, 4, 4
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.randn(n_in, m, dtype=torch.float64)
    U[:, 3] = U[:, 0]
    V[:, 3] = V[:, 0]
    s = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    t = s.clone()
    ridge, jitter = 1e-8, 1e-6

    result = logdet_greedy(s, t, U=U, V=V, K=m, ridge=ridge, jitter=jitter)
    order = result.indices.tolist()
    assert 0 in order
    if 3 in order:
        pos0, pos3 = order.index(0), order.index(3)
        assert pos0 < pos3
        # a duplicate that *is* bought must have collapsed to (near) the
        # jitter floor, never a healthy positive gain.
        assert float(result.gains[pos3]) < math.log(jitter) + 1.0
    else:
        # observed outcome for this ridge/jitter pair: once atom 0 is
        # bought, the pivoted-Cholesky update drives the twin's residual
        # variance down to ~2*ridge (well below jitter=1e-6), so index 3
        # never resurfaces at the top of argmax again.
        pass


def test_logdet_greedy_rent_stops_exactly_at_the_predicted_gain_crossing() -> None:
    # A diagonal Gram (orthogonal U/V columns with distinct scales) makes the
    # greedy order and per-step gains fully predictable: no cross terms, so
    # gains are exactly log(diag_i + ridge) in strictly decreasing order.
    m = 6
    diag_target = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0]
    U = torch.zeros(m, m, dtype=torch.float64)
    V = torch.zeros(m, m, dtype=torch.float64)
    for i, value in enumerate(diag_target):
        U[i, i] = math.sqrt(value)
        V[i, i] = 1.0
    s = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    t = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    ridge, jitter = 1e-6, 1e-9

    full = logdet_greedy(s, t, U=U, V=V, K=m, ridge=ridge, jitter=jitter)
    assert full.indices.tolist() == list(range(m))
    gains = full.gains.tolist()
    # Non-increasing by construction on this diagonal, well-separated pool --
    # NOT a general guarantee of logdet_greedy (submodularity gives
    # diminishing *set* gains, not monotone *per-step* gains in general).
    assert all(gains[i] >= gains[i + 1] - 1e-12 for i in range(len(gains) - 1))

    i = 2
    rent = math.exp((gains[i] + gains[i + 1]) / 2.0)
    capped = logdet_greedy(s, t, U=U, V=V, K=m, rent=rent, ridge=ridge, jitter=jitter)
    assert capped.indices.tolist() == list(range(i + 1))


def test_logdet_greedy_deprioritizes_data_null_space_candidates() -> None:
    torch.manual_seed(2)
    n_out, n_in = 3, 4
    base = torch.randn(200, 2, dtype=torch.float64)
    projection = torch.randn(2, n_in, dtype=torch.float64)
    data = base @ projection  # Sigma_x has rank <= 2, a 2-dim null space

    sigma = data.transpose(0, 1) @ data / data.shape[0]
    _, evecs = torch.linalg.eigh(sigma)
    null_vecs = evecs[:, :2]  # smallest-eigenvalue directions
    row_vecs = evecs[:, 2:]  # largest-eigenvalue (row-space) directions

    m = 6
    U = torch.randn(n_out, m, dtype=torch.float64)
    V = torch.zeros(n_in, m, dtype=torch.float64)
    null_indices = {0, 1, 2}
    row_indices = {3, 4, 5}
    V[:, 0] = null_vecs[:, 0]
    V[:, 1] = null_vecs[:, 1]
    V[:, 2] = (null_vecs[:, 0] + null_vecs[:, 1]) / math.sqrt(2.0)
    V[:, 3] = row_vecs[:, 0]
    V[:, 4] = row_vecs[:, 1]
    V[:, 5] = (row_vecs[:, 0] + row_vecs[:, 1]) / math.sqrt(2.0)
    s = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    t = torch.arange(m, dtype=torch.float64).unsqueeze(1)
    ridge, jitter = 1e-6, 1e-6

    result = logdet_greedy(s, t, U=U, V=V, K=m, ridge=ridge, jitter=jitter, data=data)
    order = result.indices.tolist()

    row_selected = [i for i in order if i in row_indices]
    null_selected = [i for i in order if i in null_indices]
    assert row_selected  # the data metric never starves visible directions
    if null_selected:
        last_row_pos = max(order.index(i) for i in row_selected)
        first_null_pos = min(order.index(i) for i in null_selected)
        assert last_row_pos < first_null_pos
        row_gains = [float(result.gains[order.index(i)]) for i in row_selected]
        null_gains = [float(result.gains[order.index(i)]) for i in null_selected]
        assert max(null_gains) < min(row_gains)


def test_logdet_greedy_requires_exactly_one_stopping_rule_and_one_factor_source() -> None:
    s = torch.zeros((3, 1), dtype=torch.float64)
    t = torch.zeros((3, 1), dtype=torch.float64)
    U = torch.eye(3, dtype=torch.float64)
    V = torch.eye(3, dtype=torch.float64)

    try:
        logdet_greedy(s, t, U=U, V=V, ridge=1e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError with neither K nor rent given")

    try:
        logdet_greedy(s, t, ridge=1e-6, K=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError with neither (U, V) nor (U_fn, V_fn) given")

    try:
        logdet_greedy(s, t, U=U, V=V, U_fn=lambda t: t, K=2, ridge=1e-6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when mixing (U, V) and (U_fn, V_fn)")


# ---------------------------------------------------------------------------
# variance_amplitudes
# ---------------------------------------------------------------------------


def test_variance_amplitudes_matches_target_second_moment_exactly() -> None:
    torch.manual_seed(4)
    n_out, n_in, k = 4, 3, 5
    U = torch.randn(n_out, k, dtype=torch.float64)
    V = torch.randn(n_in, k, dtype=torch.float64)
    gamma = (U.transpose(0, 1) @ U) * (V.transpose(0, 1) @ V)
    target = 2.5

    w = variance_amplitudes(
        U=U, V=V, target_second_moment=target, generator=torch.Generator().manual_seed(0)
    )
    achieved = float(w @ gamma @ w)
    torch.testing.assert_close(achieved, target, atol=1e-8, rtol=0.0)

    # sign entries are +/- the same kappa
    kappa = float(w.abs()[0])
    torch.testing.assert_close(w.abs(), torch.full_like(w, kappa), atol=1e-12, rtol=0.0)


def test_variance_amplitudes_is_deterministic_given_a_seeded_generator() -> None:
    torch.manual_seed(5)
    n_out, n_in, k = 3, 3, 4
    U = torch.randn(n_out, k, dtype=torch.float64)
    V = torch.randn(n_in, k, dtype=torch.float64)

    w1 = variance_amplitudes(
        U=U, V=V, target_second_moment=1.0, generator=torch.Generator().manual_seed(42)
    )
    w2 = variance_amplitudes(
        U=U, V=V, target_second_moment=1.0, generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(w1, w2)

    w3 = variance_amplitudes(
        U=U, V=V, target_second_moment=1.0, generator=torch.Generator().manual_seed(43)
    )
    assert not torch.equal(w1, w3)


def test_variance_amplitudes_accepts_precomputed_gamma() -> None:
    torch.manual_seed(6)
    k = 4
    a = torch.randn(k, k, dtype=torch.float64)
    gamma = a @ a.transpose(0, 1) + 1e-3 * torch.eye(k, dtype=torch.float64)
    target = 4.0

    w = variance_amplitudes(
        Gamma=gamma, target_second_moment=target, generator=torch.Generator().manual_seed(1)
    )
    achieved = float(w @ gamma @ w)
    torch.testing.assert_close(achieved, target, atol=1e-8, rtol=0.0)


# ---------------------------------------------------------------------------
# to_synapse_births
# ---------------------------------------------------------------------------


def test_to_synapse_births_wraps_one_batched_op() -> None:
    s = torch.zeros((3, 1), dtype=torch.float64)
    t = torch.zeros((3, 1), dtype=torch.float64)
    w = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)

    ops = to_synapse_births("layer", s, t, w, lineage_start=7)
    assert len(ops) == 1
    op = ops[0]
    assert op.site == "layer"
    assert torch.equal(op.s, s)
    assert torch.equal(op.t, t)
    assert torch.equal(op.w, w)
    assert op.lineage.dtype == torch.int64
    assert op.lineage.tolist() == [7, 8, 9]


# ---------------------------------------------------------------------------
# Integration: coverage_lattice + variance_amplitudes -> CSTLinear -> GramService
# ---------------------------------------------------------------------------


def test_coverage_init_keeps_local_gram_well_conditioned_end_to_end() -> None:
    torch.manual_seed(0)
    n_in, n_out = 6, 5
    sigma = 0.3
    inputs = NeuronStore(
        "in",
        n_in,
        mu=torch.linspace(-1.0, 1.0, n_in, dtype=torch.float64)[:, None],
        initial_live=n_in,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "out",
        n_out,
        mu=torch.linspace(-1.0, 1.0, n_out, dtype=torch.float64)[:, None],
        initial_live=n_out,
        dtype=torch.float64,
    )
    synapses = SynapseStore(
        "layer",
        d_in=1,
        d_out=1,
        capacity=128,
        spec=RepresentationSpec.continuous(1, 1, bounds=(-1.0, 1.0)),
        dtype=torch.float64,
    )
    layer = CSTLinear(inputs, outputs, synapses, GaussianFactor(sigma))

    s, t = coverage_lattice((-1.0, 1.0), (-1.0, 1.0), 1, 1, sigma, sigma, spacing=1.0)
    s = s.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64)
    # factor_columns returns (k_in, k_out): k_in is the input-side (V)
    # column [n_in, M], k_out is the output-side (U) column [n_out, M] --
    # GramService's own naming convention.
    k_in, k_out = layer.factor_columns(s, t)
    w = variance_amplitudes(
        U=k_out, V=k_in, target_second_moment=1.0, generator=torch.Generator().manual_seed(0)
    )
    synapses.apply(to_synapse_births("layer", s, t, w))

    x = torch.randn(8, n_in, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (8, n_out)
    assert bool(torch.isfinite(y).all())

    view = synapses.view()
    k_in_live, k_out_live = layer.factor_columns(view.s, view.t)
    gram = GramService(
        k_out_live, k_in_live, view.w, view.s, view.t, radius=1.2 * sigma, ridge=1e-6
    )
    n_atoms = view.ids.numel()
    lambdas = [gram.local_lambda_min(k) for k in range(n_atoms)]
    assert min(lambdas) > 1e-3
