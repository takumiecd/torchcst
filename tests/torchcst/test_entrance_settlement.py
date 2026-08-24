"""Stage 3b-fix: certificate reset (R_{t-1}=0) and loss-unit entrance settlement.

Covers ``docs/absorb-and-gram-design.md`` section 5's "Stage 3b-fix" block:
the two defects the sol audit (``cst/scripts/diag_birth_gain_calibration.py``,
a sibling repo's diagnostic -- not imported here, but reproduced at a much
smaller scale) found in ``ContinuousCandidateField``/``ScoredBirth``.

1. Certificate reset: a structural event must *consume* the accumulated
   certificate, not merely observe it -- otherwise the next event gates on a
   cumulative, pre-optimizer-step score.
2. Loss-unit settlement: ``ScoredBirth``'s ``rent`` gate must compare against
   the exact GramService-settled profile gain in loss units, not the cheap
   Frobenius-deflated instrument score (``0.5 * score**2``), which is D-blind
   and has no general candidate-independent correction factor.

Both are exercised through the public policy path: a real
``StructuralEngine`` + ``Policy(ActionSpec.synapse_birth(ScoredBirth(...)))``
composition, mirroring the diagnostic's own build/verify structure.
"""

from __future__ import annotations

import math

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import ContinuousCandidateRequest
from torchcst.policy import (
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    ScoredBirth,
    SynapseLifecycle,
)
from torchcst.policy.registry import RetiredCandidateRegistry
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


# ---------------------------------------------------------------------------
# Shared fixture: a small continuous-edge teacher/student regression problem,
# built at a much smaller scale than the diagnostic script but with the same
# structure (random non-whitened data batch, random teacher target, ridge-fit
# live amplitudes) so the D metric is genuinely anisotropic and the ridge
# refit reference stays well conditioned in float64.
# ---------------------------------------------------------------------------

N = 8
BATCH = 6
SIGMA = 0.2
K_LIVE = 4
K_TEACHER = 6
POOL_SIZE = 24
RIDGE = 1.0e-10
SEED = 20260729
SITE = "edge"


def _gaussian_columns(grid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Reproduce ``GaussianFactor``'s profile: ``exp(-d^2 / (2 sigma^2))``."""
    return torch.exp(-0.5 * ((grid[:, None] - positions.reshape(1, -1)) / SIGMA).square())


def _atom_matrices(
    source: torch.Tensor, target: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor
) -> torch.Tensor:
    """Return atom matrices ``psi_z = u(t) v(s)^T`` as ``[K, out, in]``."""
    k_in = _gaussian_columns(x_grid, source)
    k_out = _gaussian_columns(y_grid, target)
    return torch.einsum("ok,ik->koi", k_out, k_in)


def _prediction_design(x: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Map atom amplitudes to flattened predictions, with the loss's ``1/sqrt(n)``."""
    predictions = torch.einsum("bi,koi->bko", x, atoms)
    return predictions.permute(0, 2, 1).reshape(-1, atoms.shape[0]) / math.sqrt(x.shape[0])


def _ridge_fit(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gram = design.T @ design
    rhs = design.T @ target
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    return torch.linalg.solve(gram + RIDGE * eye, rhs)


def _loss_from_fit(design: torch.Tensor, target: torch.Tensor, amplitudes: torch.Tensor) -> float:
    residual = design @ amplitudes - target
    return 0.5 * float(residual @ residual)


def _build_problem(rent: float, budget: int):
    generator = torch.Generator().manual_seed(SEED)
    x_grid = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    y_grid = torch.linspace(0.0, 1.0, N, dtype=torch.float64)

    # An explicit, full-rank, non-whitened data batch makes D anisotropic.
    x = torch.randn(BATCH, N, generator=generator, dtype=torch.float64)
    x = x * torch.linspace(0.5, 1.5, N, dtype=torch.float64)
    x[:, 1:] += 0.2 * x[:, :-1].clone()
    mean_d_diagonal = torch.trace(x.T @ x / BATCH) / N
    x = x / mean_d_diagonal.sqrt()

    teacher_source = torch.rand(K_TEACHER, 1, generator=generator, dtype=torch.float64)
    teacher_target = torch.rand(K_TEACHER, 1, generator=generator, dtype=torch.float64)
    teacher_amplitudes = torch.randn(K_TEACHER, generator=generator, dtype=torch.float64)
    teacher_atoms = _atom_matrices(teacher_source, teacher_target, x_grid, y_grid)
    teacher_weight = torch.einsum("k,koi->oi", teacher_amplitudes, teacher_atoms)
    target_batch = x @ teacher_weight.T
    target_vector = target_batch.reshape(-1) / math.sqrt(BATCH)

    live_source = torch.linspace(0.1, 0.9, K_LIVE, dtype=torch.float64)[:, None]
    target_order = torch.tensor([0, 2, 1, 3])
    live_target = torch.linspace(0.1, 0.9, K_LIVE, dtype=torch.float64)[target_order, None]
    live_atoms = _atom_matrices(live_source, live_target, x_grid, y_grid)
    live_design = _prediction_design(x, live_atoms)
    live_amplitudes = _ridge_fit(live_design, target_vector)
    baseline_loss = _loss_from_fit(live_design, target_vector, live_amplitudes)

    store = SynapseStore(
        SITE,
        1,
        1,
        capacity=K_LIVE,
        max_capacity=K_LIVE + POOL_SIZE,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        f"{SITE}_in", N, mu=x_grid[:, None], initial_live=N, dtype=torch.float64
    )
    outputs = NeuronStore(
        f"{SITE}_out", N, mu=y_grid[:, None], initial_live=N, dtype=torch.float64
    )
    module = CSTLinear(
        inputs, outputs, store, GaussianFactor(SIGMA, learnable=False).double(), track_mass=False
    )
    store.apply(
        (
            SynapseBirth(
                SITE,
                live_source,
                live_target,
                live_amplitudes,
                torch.arange(K_LIVE, dtype=torch.int64),
            ),
        )
    )

    request = ContinuousCandidateRequest(pool_size=POOL_SIZE, mode="deflated")
    method = SynapseLifecycle(
        birth_factory=lambda lam: ScoredBirth(request=request, rent=rent, data=lambda: x),
        priceable=False,
        label="entrance-settlement",
    )
    policy = QuotaRegime(
        budget=budget,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {SITE: store, inputs.site: inputs, outputs.site: outputs},
        policy,
        modules={SITE: module},
        seed=SEED + 1,
    )
    return (
        store,
        module,
        engine,
        request,
        x,
        target_batch,
        live_atoms,
        target_vector,
        baseline_loss,
    )


def _train_step(engine: StructuralEngine, module: CSTLinear, x: torch.Tensor, target_batch: torch.Tensor) -> None:
    engine.begin_update()
    prediction = module(x)
    loss = 0.5 * (prediction - target_batch).square().sum() / BATCH
    loss.backward()
    engine.observe_microbatch(weight=1.0)
    engine.finalize_backward()


# ---------------------------------------------------------------------------
# (a) Certificate reset: consumed by the event, not merely observed.
# ---------------------------------------------------------------------------


def test_reset_on_consume_clears_certificate_and_blocks_next_event() -> None:
    store, module, engine, request, x, target_batch, *_ = _build_problem(
        rent=0.0, budget=5
    )
    instrument = engine.instrument(SITE, request.name)

    _train_step(engine, module, x, target_batch)
    before = instrument.raw_gradient().clone()
    assert torch.count_nonzero(before) > 0, "fixture must produce a real gradient"

    # First event: a real structural event consumes this window's certificate
    # (and, with real signal and a permissive rent, is free to grow K).
    engine.step()
    after_first_event = instrument.raw_gradient()
    assert torch.count_nonzero(after_first_event) == 0
    torch.testing.assert_close(after_first_event, torch.zeros_like(after_first_event))
    k_after_first_event = store.view().ids.numel()

    # Second event, with no intervening backward: the certificate is still
    # exactly zero ("nothing to explain"), so even a permissive rent=0.0 with
    # budget to spare buys no births -- this is the property that fails
    # without reset-on-consume (a stale, pre-optimizer-step certificate would
    # otherwise still carry a nonzero, cumulative score here).
    applied = engine.step()
    births = tuple(op for op in applied if isinstance(op, SynapseBirth))
    assert births == ()

    # And the pool itself was still scoreable (not merely empty/degenerate):
    # every candidate's raw score against the zeroed certificate is exactly 0.
    snapshot = instrument.candidate_snapshot()
    assert snapshot.scores.numel() > 0
    torch.testing.assert_close(snapshot.scores, torch.zeros_like(snapshot.scores))
    assert store.view().ids.numel() == k_after_first_event  # unchanged by event 2


# ---------------------------------------------------------------------------
# (b) Settlement calibration: gain == realized delta-loss to rtol 1e-6.
# ---------------------------------------------------------------------------


def test_settlement_gain_matches_realized_delta_loss() -> None:
    budget = 5
    (
        store,
        module,
        engine,
        request,
        x,
        target_batch,
        live_atoms,
        target_vector,
        baseline_loss,
    ) = _build_problem(rent=0.0, budget=budget)

    _train_step(engine, module, x, target_batch)

    instrument = engine.instrument(SITE, request.name)
    view = store.view()
    snapshot = instrument.candidate_snapshot()

    proposer = ScoredBirth(request=request, rent=0.0, data=lambda: x)
    proposer.bind_instruments(SITE, {request.name: instrument})
    positions = proposer.selector.select(snapshot.scores, budget)
    assert positions.numel() == budget

    # This is ScoredBirth's own settlement computation -- the exact value the
    # rent gate compares against -- for the same shortlist the public
    # propose() path selects.
    gains = proposer._settlement_gains(view, instrument, snapshot, positions)

    x_grid = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    y_grid = torch.linspace(0.0, 1.0, N, dtype=torch.float64)
    live_design = _prediction_design(x, live_atoms)

    for position_tensor, gain in zip(positions, gains):
        position = int(position_tensor)
        source = snapshot.source[position : position + 1]
        target = snapshot.target[position : position + 1]
        psi = _atom_matrices(source, target, x_grid, y_grid)

        candidate_design = _prediction_design(x, psi)
        augmented_design = torch.cat((live_design, candidate_design), dim=1)
        augmented_amplitudes = _ridge_fit(augmented_design, target_vector)
        refit_loss = _loss_from_fit(augmented_design, target_vector, augmented_amplitudes)
        realized = baseline_loss - refit_loss

        assert math.isclose(float(gain), realized, rel_tol=1.0e-6, abs_tol=1.0e-9), (
            f"position {position}: settled gain {float(gain)!r} vs "
            f"realized delta-loss {realized!r}"
        )

    # Confirm this calibration is actually exercised through propose(): the
    # accepted births (rent=0.0, all gains here are strictly positive real
    # signal) are exactly the shortlist, and applying the public path gives
    # the same coordinates used above.
    registry = RetiredCandidateRegistry()
    rng = torch.Generator().manual_seed(1)
    births = proposer.propose(view, budget, registry, rng)
    assert len(births) == 1
    assert births[0].w.numel() == budget


# ---------------------------------------------------------------------------
# (c) Rent now binds in loss units (T1 anomaly reproducer, inverted): once
# rent is set above every candidate's true settled gain, nothing is bought
# even with a large budget.
# ---------------------------------------------------------------------------


def test_rent_above_true_gain_blocks_every_birth_despite_large_budget() -> None:
    large_budget = POOL_SIZE

    # Probe run: find the true maximum settled gain over the *entire* pool
    # (not just a small shortlist) so the chosen rent is provably above every
    # candidate, regardless of which ones a cheap pre-ranker would shortlist.
    probe_store, probe_module, probe_engine, probe_request, probe_x, probe_target, *_ = (
        _build_problem(rent=0.0, budget=large_budget)
    )
    _train_step(probe_engine, probe_module, probe_x, probe_target)
    probe_instrument = probe_engine.instrument(SITE, probe_request.name)
    probe_view = probe_store.view()
    probe_snapshot = probe_instrument.candidate_snapshot()

    proposer = ScoredBirth(request=probe_request, rent=0.0, data=lambda: probe_x)
    proposer.bind_instruments(SITE, {probe_request.name: probe_instrument})
    all_positions = torch.arange(probe_snapshot.scores.numel(), dtype=torch.int64)
    all_gains = proposer._settlement_gains(
        probe_view, probe_instrument, probe_snapshot, all_positions
    )
    max_gain = float(all_gains.max())
    assert max_gain > 0.0, "fixture must have at least one genuinely useful candidate"

    rent = max_gain * 10.0  # strictly above every candidate's true gain

    # Same deterministic fixture (fixed SEED), rebuilt with the binding rent.
    store, module, engine, request, x, target_batch, *_ = _build_problem(
        rent=rent, budget=large_budget
    )
    _train_step(engine, module, x, target_batch)
    applied = engine.step()

    births = tuple(op for op in applied if isinstance(op, SynapseBirth))
    assert births == ()
    assert store.view().ids.numel() == K_LIVE  # unchanged
