"""GramService: separable Gram assembly, local ridge absorb, and chain planning."""

from __future__ import annotations

import math

import torch

from torchcst.compute import RankOneLinear
from torchcst.representation import GramService, RepresentationSpec
from torchcst.storage import SynapseAbsorb, SynapseBirth, SynapseStore


def _represented_matrix(U: torch.Tensor, V: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``sum_k w_k * outer(U[:, k], V[:, k])``, shape ``[n_out, n_in]``."""
    return (U * w.unsqueeze(0)) @ V.transpose(0, 1)


def _direct_gram(
    U: torch.Tensor, V: torch.Tensor, idx: torch.Tensor, sigma: torch.Tensor | None
) -> torch.Tensor:
    """Dense ``<psi_i, psi_j>`` (D-metric if sigma given, else Frobenius)."""
    n = idx.numel()
    out = torch.zeros((n, n), dtype=U.dtype, device=U.device)
    for a, i in enumerate(idx.tolist()):
        psi_i = torch.outer(U[:, i], V[:, i])
        for b, j in enumerate(idx.tolist()):
            psi_j = torch.outer(U[:, j], V[:, j])
            if sigma is None:
                out[a, b] = (psi_i * psi_j).sum()
            else:
                out[a, b] = torch.trace(psi_i @ sigma @ psi_j.transpose(0, 1))
    return out


def _placed(k: int, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, well-separated (s, t) placements so radius screening is trivial."""
    s = torch.full((dim,), float(10 * k), dtype=torch.float64)
    t = torch.full((dim,), float(10 * k), dtype=torch.float64)
    return s, t


def test_gram_block_matches_dense_direct_computation_identity_and_data_metric() -> None:
    torch.manual_seed(0)
    n_out, n_in, k_live = 4, 3, 5
    U = torch.randn(n_out, k_live, dtype=torch.float64)
    V = torch.randn(n_in, k_live, dtype=torch.float64)
    w = torch.randn(k_live, dtype=torch.float64)
    s = torch.stack([_placed(i)[0] for i in range(k_live)])
    t = torch.stack([_placed(i)[1] for i in range(k_live)])
    idx = torch.arange(k_live, dtype=torch.int64)

    identity_gram = GramService(U, V, w, s, t, radius=1000.0, ridge=1e-6)
    gamma_d, gamma_f = identity_gram.gram_block(idx, idx)
    torch.testing.assert_close(gamma_d, _direct_gram(U, V, idx, sigma=None))
    torch.testing.assert_close(gamma_f, _direct_gram(U, V, idx, sigma=None))

    n = 200
    data = torch.randn(n, n_in, dtype=torch.float64)
    sigma = data.transpose(0, 1) @ data / n
    data_gram = GramService(U, V, w, s, t, radius=1000.0, ridge=1e-6, data=data)
    gamma_d2, gamma_f2 = data_gram.gram_block(idx, idx)
    torch.testing.assert_close(gamma_d2, _direct_gram(U, V, idx, sigma=sigma))
    torch.testing.assert_close(gamma_f2, _direct_gram(U, V, idx, sigma=None))


def test_planted_exact_duplicate_gives_near_zero_residual_and_concentrated_alpha() -> None:
    # Atom 0 is the target; atom 1 is its exact twin (same U/V column, placed
    # close by); atom 2 is unrelated and placed far outside the radius.
    u0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    v0 = torch.tensor([0.6, 0.8], dtype=torch.float64)
    U = torch.stack([u0, u0, torch.tensor([0.0, 1.0], dtype=torch.float64)], dim=1)
    V = torch.stack([v0, v0, torch.tensor([1.0, 0.0], dtype=torch.float64)], dim=1)
    w = torch.tensor([1.25, -0.5, 3.0], dtype=torch.float64)
    s = torch.tensor([[0.0], [0.01], [500.0]], dtype=torch.float64)
    t = torch.tensor([[0.0], [0.01], [500.0]], dtype=torch.float64)

    gram = GramService(U, V, w, s, t, radius=1.0, ridge=1e-10)
    neigh = gram.neighbors(0)
    assert neigh.tolist() == [1]

    assessment = gram.residual(0)
    assert assessment.res2_D < 1e-12
    assert assessment.receivers.tolist() == [1]
    torch.testing.assert_close(
        assessment.alpha, torch.tensor([1.0], dtype=torch.float64), atol=1e-4, rtol=0.0
    )


def test_planted_oversampled_cluster_hides_from_pairwise_coherence() -> None:
    # Three atoms share the same output-side factor (t0), so Gamma_ij ==
    # (s_i . s_j) exactly once normalized (unit s vectors). s2 is the exact
    # unit-norm combination of s0/s1, so psi_2 == psi_0/sqrt(2) + psi_1/sqrt(2)
    # exactly, but every pairwise coherence stays well below a duplicate
    # threshold that would flag e.g. cos >= 0.9.
    t0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    s0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    s1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    s2 = (s0 + s1) / math.sqrt(2.0)
    U = torch.stack([t0, t0, t0], dim=1)
    V = torch.stack([s0, s1, s2], dim=1)
    w = torch.tensor([0.9, -1.1, 2.0], dtype=torch.float64)
    s = torch.stack([_placed(i, dim=1)[0] for i in range(3)])
    t = torch.stack([_placed(i, dim=1)[1] for i in range(3)])

    gram = GramService(U, V, w, s, t, radius=1000.0, ridge=1e-10)
    idx = torch.arange(3, dtype=torch.int64)
    gamma_d, _ = gram.gram_block(idx, idx)
    diag = torch.diagonal(gamma_d)
    coherence = gamma_d / torch.outer(diag.sqrt(), diag.sqrt())
    off_diag = coherence[~torch.eye(3, dtype=torch.bool)]
    duplicate_threshold = 0.9
    assert bool((off_diag.abs() < duplicate_threshold).all())

    assessment = gram.residual(2)
    assert assessment.res2_D < 1e-12
    assert assessment.receivers.numel() == 2
    assert torch.all(assessment.alpha.abs() > 0.1)
    expected = 1.0 / math.sqrt(2.0)
    torch.testing.assert_close(
        assessment.alpha.sort().values,
        torch.tensor([expected, expected], dtype=torch.float64),
        atol=1e-4,
        rtol=0.0,
    )


def test_isolated_atom_pays_its_full_norm_with_empty_alpha() -> None:
    u0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    v0 = torch.tensor([0.6, 0.8], dtype=torch.float64)
    U = torch.stack([u0, torch.tensor([0.0, 1.0], dtype=torch.float64)], dim=1)
    V = torch.stack([v0, torch.tensor([1.0, 0.0], dtype=torch.float64)], dim=1)
    w = torch.tensor([1.25, 3.0], dtype=torch.float64)
    s = torch.tensor([[0.0], [500.0]], dtype=torch.float64)
    t = torch.tensor([[0.0], [500.0]], dtype=torch.float64)

    gram = GramService(U, V, w, s, t, radius=1.0, ridge=1e-8)
    assert gram.neighbors(0).numel() == 0

    assessment = gram.residual(0)
    assert assessment.alpha.numel() == 0
    assert assessment.receivers.numel() == 0
    expected_norm = float((torch.outer(u0, v0) ** 2).sum())
    torch.testing.assert_close(assessment.res2_D, expected_norm)


def test_ridge_bounds_alpha_on_near_singular_cluster_unridged_blows_up() -> None:
    # Two neighbors are nearly parallel to each other (a near-singular 2x2
    # local Gram: eigenvalues ~4e-8 and ~2), and the target's u/v factors are
    # deliberately *not* equal, so its overlap with the neighbors' near-null
    # (antisymmetric) eigenvector is a real, non-vanishing O(eps) signal
    # rather than the O(eps^2) that a symmetric (duplicate-shaped) target
    # would give -- that is what makes alpha explode as ridge -> 0 instead
    # of settling on a bounded min-norm solution. A healthy ridge keeps
    # ||alpha|| modest; a near-zero ridge does not (demonstrated only; a
    # near-zero ridge is never what a caller should actually use).
    u0 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    v0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    eps = 1e-4
    U = torch.stack(
        [
            u0,
            torch.tensor([1.0, eps], dtype=torch.float64),
            torch.tensor([1.0, -eps], dtype=torch.float64),
        ],
        dim=1,
    )
    V = torch.stack(
        [
            v0,
            torch.tensor([1.0, eps], dtype=torch.float64),
            torch.tensor([1.0, -eps], dtype=torch.float64),
        ],
        dim=1,
    )
    w = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    s = torch.stack([_placed(i, dim=1)[0] for i in range(3)])
    t = torch.stack([_placed(i, dim=1)[1] for i in range(3)])

    healthy = GramService(U, V, w, s, t, radius=1000.0, ridge=1e-2)
    fragile = GramService(U, V, w, s, t, radius=1000.0, ridge=1e-16)

    healthy_alpha = healthy.residual(0).alpha
    fragile_alpha = fragile.residual(0).alpha
    assert float(torch.linalg.vector_norm(healthy_alpha)) < 1.0
    assert float(torch.linalg.vector_norm(fragile_alpha)) > 1.0e3


def test_absorb_cost_identity_matches_quadratic_loss_increase() -> None:
    torch.manual_seed(1)
    d_in, d_out, k_live = 3, 2, 3
    source = torch.randn(k_live, d_in, dtype=torch.float64)
    source = source / torch.linalg.vector_norm(source, dim=1, keepdim=True)
    target = torch.randn(k_live, d_out, dtype=torch.float64)
    target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True)
    weights = torch.tensor([1.1, -0.7, 0.4], dtype=torch.float64)

    store = SynapseStore(
        "rank",
        d_in,
        d_out,
        k_live,
        spec=RepresentationSpec.rank_one(d_in, d_out),
        dtype=torch.float64,
    )
    store.apply(
        [SynapseBirth("rank", source, target, weights, torch.arange(k_live, dtype=torch.int64))]
    )
    module = RankOneLinear(store, d_in, d_out)
    w_true = module.dense_weight().detach().clone()  # baseline loss is exactly zero

    n = 500
    data = torch.randn(n, d_in, dtype=torch.float64)
    sigma = data.transpose(0, 1) @ data / n

    view = store.view()
    U = view.t.detach().transpose(0, 1).clone()
    V = view.s.detach().transpose(0, 1).clone()
    w = view.w.detach().clone()
    gram = GramService(
        U, V, w, view.s.detach(), view.t.detach(), radius=1000.0, ridge=1e-3, data=data
    )

    k = 0
    assessment = gram.residual(k)
    c = float(w[k])
    cost = gram.cost(k)
    torch.testing.assert_close(cost, 0.5 * c**2 * assessment.res2_D)

    dying_id = int(view.ids[k])
    receiver_ids = view.ids.index_select(0, assessment.receivers)
    store.apply(
        [
            SynapseAbsorb(
                "rank",
                dying_id,
                receiver_ids.clone(),
                (c * assessment.alpha).clone(),
            )
        ]
    )

    w_after = module.dense_weight().detach()
    delta = w_after - w_true
    loss_after = 0.5 * torch.trace(delta @ sigma @ delta.transpose(0, 1))
    torch.testing.assert_close(float(loss_after), cost, rtol=1e-6, atol=1e-9)


def test_local_lambda_min_flags_fiber_aligned_set_pairwise_screening_would_pass() -> None:
    # Five atoms share the same output-side factor u0 (a "fiber"); their
    # input-side factors are unit vectors spread 72 degrees apart in R^2
    # ("spread s", z-distances large by construction below). Because they
    # all share u0, psi_i = outer(u0, v_i) lives in a subspace isomorphic to
    # the 2-D v-space, so any 3+ of the 5 atoms are exactly linearly
    # dependent -- yet every pairwise coherence (|cos(72 deg)| ~= 0.309,
    # |cos(144 deg)| ~= 0.809) stays under a 0.85 duplicate threshold.
    u0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    angles = [2 * math.pi * i / 5 for i in range(5)]
    V = torch.stack(
        [torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float64) for a in angles],
        dim=1,
    )
    U = u0.unsqueeze(1).repeat(1, 5)
    w = torch.ones(5, dtype=torch.float64)
    # Spread s widely (large pairwise z-distance); t identical (the fiber).
    s = 5.0 * V.transpose(0, 1).clone()
    t = torch.zeros((5, 2), dtype=torch.float64)

    gram = GramService(U, V, w, s, t, radius=100.0, ridge=1e-10)
    idx = torch.arange(5, dtype=torch.int64)
    gamma_d, _ = gram.gram_block(idx, idx)
    diag = torch.diagonal(gamma_d)
    coherence = gamma_d / torch.outer(diag.sqrt(), diag.sqrt())
    off_diag = coherence[~torch.eye(5, dtype=torch.bool)]
    assert bool((off_diag.abs() < 0.85).all())

    assert gram.neighbors(0).numel() == 4  # all pass the (generous) radius
    assert gram.local_lambda_min(0) < 1e-8


def test_plan_chain_on_m_copy_cluster_leaves_one_survivor_near_zero_displacement() -> None:
    m = 8
    u0 = torch.tensor([0.6, 0.8], dtype=torch.float64)
    v0 = torch.tensor([0.8, 0.6], dtype=torch.float64)
    U = u0.unsqueeze(1).repeat(1, m)
    V = v0.unsqueeze(1).repeat(1, m)
    w = torch.linspace(0.5, 0.5 * m, m, dtype=torch.float64)
    s = torch.zeros((m, 1), dtype=torch.float64)
    t = torch.zeros((m, 1), dtype=torch.float64)

    gram = GramService(U, V, w, s, t, radius=1.0, ridge=1e-9)
    steps = gram.plan_chain()

    assert len(steps) == m - 1
    for step in steps:
        assert step.cost < 1e-6

    running_w = w.clone()
    for step in steps:
        running_w[step.receivers] += step.delta_w
        running_w[step.dying] = 0.0
    survivors = torch.nonzero(running_w != 0.0, as_tuple=False).flatten()
    assert survivors.numel() == 1

    w_before = _represented_matrix(U, V, w)
    w_after = _represented_matrix(U, V, running_w)
    displacement = float(torch.linalg.matrix_norm(w_after - w_before))
    scale = float(torch.linalg.matrix_norm(w_before))
    assert displacement < 1e-4 * scale
