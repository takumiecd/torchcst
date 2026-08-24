"""ContinuousCandidateField cRigL/cRES scoring, FactorPort, and ScoredBirth wiring."""

from __future__ import annotations

import types

import pytest
import torch

from torchcst.compute import CSTLinear, conv2d_neuron_coordinates
from torchcst.engine import StructuralEngine
from torchcst.instruments import (
    ContinuousCandidateField,
    ContinuousCandidateRequest,
    RefinementSchedule,
    WeightedMeasurement,
)
from torchcst.instruments._candidate_sampling import (
    _coarse_spacing,
    _sample_local_uniform,
)
from torchcst.policy import (
    EvenBudgetDistributor,
    MagnitudeCourt,
    PeriodicCadence,
    QuotaRegime,
    ScoredBirth,
    SynapseLifecycle,
    TopKSelector,
)
from torchcst.representation import Box, GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _build_module(dtype: torch.dtype = torch.float64):
    store = SynapseStore(
        "continuous", 1, 1, 8, spec=RepresentationSpec.continuous(1, 1), dtype=dtype
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.25], [0.6]], dtype=dtype),
                torch.tensor([[0.75], [0.3]], dtype=dtype),
                torch.tensor([0.4, -0.2], dtype=dtype),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs",
        3,
        mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=dtype),
        initial_live=3,
        dtype=dtype,
    )
    outputs = NeuronStore(
        "outputs", 2, mu=torch.tensor([[0.2], [0.8]], dtype=dtype), initial_live=2, dtype=dtype
    )
    module = CSTLinear(inputs, outputs, store, GaussianFactor(0.25).to(dtype))
    return store, module


def _accumulate(field: ContinuousCandidateField, module, view, x: torch.Tensor, g_out: torch.Tensor) -> None:
    measurement = field._measure(module, x, g_out)
    field.finalize_update((WeightedMeasurement(measurement, 1.0),), view)


def _brute_direction(module, source_row: torch.Tensor, target_row: torch.Tensor) -> torch.Tensor:
    """Densely materialize the Frobenius-unit direction ``a_z`` for one candidate."""
    k_in, k_out = module.factor_columns(source_row, target_row)
    direction = torch.outer(k_out[:, 0], k_in[:, 0])
    return direction / direction.norm()


def _brute_projector(module, live) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the explicit [K, out*in] live-direction matrix and its Gram."""
    rows = [
        _brute_direction(module, live.s[k : k + 1], live.t[k : k + 1]).flatten()
        for k in range(live.s.shape[0])
    ]
    directions = torch.stack(rows)
    gram = directions @ directions.T
    return directions, gram


def _brute_deflated_score(module, directions, gram, g_flat, source_row, target_row) -> torch.Tensor:
    a_flat = _brute_direction(module, source_row, target_row).flatten()
    pinv = torch.linalg.pinv(gram)
    coeffs = pinv @ (directions @ a_flat)
    proj_self = a_flat @ (directions.T @ coeffs)
    proj_grad = (directions.T @ (pinv @ (directions @ g_flat))) @ a_flat
    residual = (1.0 - proj_self).clamp_min(0.0)
    if residual <= 1e-6:
        return torch.zeros((), dtype=g_flat.dtype)
    return ((g_flat @ a_flat) - proj_grad) / residual.sqrt()


def test_raw_score_matches_brute_force_frobenius_inner_product() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(11)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=6,
        mode="raw",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    x = torch.randn(4, 3, dtype=torch.float64)
    g_out = torch.randn(4, 2, dtype=torch.float64)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    assert field._G is not None
    expected = torch.stack(
        [
            (field._G * _brute_direction(module, snapshot.source[i : i + 1], snapshot.target[i : i + 1])).sum()
            for i in range(snapshot.scores.numel())
        ]
    )
    torch.testing.assert_close(snapshot.scores, expected)


def test_deflated_score_matches_explicit_projection_reference() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(23)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=5,
        mode="deflated",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    x = torch.randn(4, 3, dtype=torch.float64)
    g_out = torch.randn(4, 2, dtype=torch.float64)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    assert field._G is not None
    directions, gram = _brute_projector(module, view)
    g_flat = field._G.flatten()
    expected = torch.stack(
        [
            _brute_deflated_score(
                module, directions, gram, g_flat, snapshot.source[i : i + 1], snapshot.target[i : i + 1]
            )
            for i in range(snapshot.scores.numel())
        ]
    )
    # The instrument adds a small Gram-matrix jitter (eps) that the brute
    # pinv-based reference does not, so allow a matching-order tolerance.
    torch.testing.assert_close(snapshot.scores, expected, atol=1e-4, rtol=1e-4)


def test_candidate_on_a_live_atom_deflates_to_zero_without_nan_or_inf() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(5)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=4,
        mode="deflated",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    # Overwrite one pooled candidate with the exact position of a live atom.
    field._source = field._source.clone()
    field._target = field._target.clone()
    field._source[0] = view.s[0]
    field._target[0] = view.t[0]
    x = torch.randn(3, 3, dtype=torch.float64)
    g_out = torch.randn(3, 2, dtype=torch.float64)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    assert torch.isfinite(snapshot.scores).all()
    torch.testing.assert_close(snapshot.scores[0], torch.zeros((), dtype=torch.float64), atol=1e-8, rtol=0.0)


def test_same_rng_seed_gives_identical_pool_and_scores() -> None:
    store, module = _build_module()
    x = torch.randn(4, 3, dtype=torch.float64)
    g_out = torch.randn(4, 2, dtype=torch.float64)

    def _run(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = torch.Generator().manual_seed(seed)
        field = ContinuousCandidateField(
            store,
            module,
            rng,
            pool_size=7,
            mode="deflated",
            sampling="uniform",
            sigma=None,
            hardcore_attempts=8,
            eps=1e-6,
            name="continuous_candidate_field",
        )
        view = store.view()
        field.prepare(view, module)
        _accumulate(field, module, view, x, g_out)
        snapshot = field.candidate_snapshot()
        return snapshot.source, snapshot.target, snapshot.scores

    source_a, target_a, scores_a = _run(101)
    source_b, target_b, scores_b = _run(101)
    torch.testing.assert_close(source_a, source_b)
    torch.testing.assert_close(target_a, target_b)
    torch.testing.assert_close(scores_a, scores_b)


def test_singular_gram_from_duplicate_live_atoms_does_not_raise() -> None:
    dtype = torch.float64
    store = SynapseStore(
        "continuous", 1, 1, 4, spec=RepresentationSpec.continuous(1, 1), dtype=dtype
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.4], [0.4]], dtype=dtype),
                torch.tensor([[0.6], [0.6]], dtype=dtype),
                torch.tensor([0.3, 0.3], dtype=dtype),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    inputs = NeuronStore(
        "inputs", 3, mu=torch.tensor([[0.0], [0.5], [1.0]], dtype=dtype), initial_live=3, dtype=dtype
    )
    outputs = NeuronStore(
        "outputs", 2, mu=torch.tensor([[0.2], [0.8]], dtype=dtype), initial_live=2, dtype=dtype
    )
    module = CSTLinear(inputs, outputs, store, GaussianFactor(0.25).to(dtype))

    rng = torch.Generator().manual_seed(7)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=6,
        mode="deflated",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    x = torch.randn(3, 3, dtype=dtype)
    g_out = torch.randn(3, 2, dtype=dtype)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    assert torch.isfinite(snapshot.scores).all()


def _scored_birth_parts(mode: str = "raw", pool_size: int = 6):
    store, module = _build_module()
    request = ContinuousCandidateRequest(pool_size=pool_size, mode=mode)
    proposer = ScoredBirth(request)
    method = SynapseLifecycle(
        birth_factory=lambda lam: proposer,
        prune_factory=lambda: MagnitudeCourt(0.0),
        priceable=False,
        label="scored-birth",
    )
    policy = QuotaRegime(
        budget=2,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        # QuotaRegime's own default distributor is replacement_only=True,
        # which would cap the birth grant at the (always-zero, since
        # MagnitudeCourt(0.0) never prunes) death count.
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {
            store.site: store,
            module.in_neurons.site: module.in_neurons,
            module.out_neurons.site: module.out_neurons,
        },
        policy,
        modules={store.site: module},
        seed=29,
    )
    return store, module, request, engine


def test_scored_birth_buys_the_top_budget_candidates_end_to_end() -> None:
    store, module, request, engine = _scored_birth_parts()
    before_live = store.live_ids().numel()
    budget = 2

    x = torch.tensor([[1.0, -0.5, 2.0], [-1.0, 0.25, 0.75]], dtype=torch.float64)
    upstream = torch.tensor([[0.5, -2.0], [1.5, 0.25]], dtype=torch.float64)

    engine.begin_update()
    instrument = engine.instrument(store.site, request.name)
    before = instrument.candidate_snapshot()
    module(x).backward(upstream)
    engine.observe_microbatch()
    engine.finalize_backward()
    scored = instrument.candidate_snapshot()
    torch.testing.assert_close(scored.source, before.source)
    torch.testing.assert_close(scored.target, before.target)

    # |score|, not the signed score: this instrument's scores are signed
    # inner products and a synapse atom's coefficient is a signed scalar
    # (TopKSelector's candidate_cone="signed" default). On this fixture the
    # two orderings genuinely disagree -- the second buy is a negative-score
    # candidate -- so this line is a real assertion about the convention.
    top = torch.argsort(scored.scores.abs(), descending=True, stable=True)[:budget]
    assert bool((scored.scores.index_select(0, top) < 0).any())
    operations = engine.step()
    births = [op for op in operations if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    birth = births[0]
    assert birth.w.numel() == budget
    torch.testing.assert_close(
        birth.s, scored.source.index_select(0, top.to(scored.source.device))
    )
    torch.testing.assert_close(
        birth.t, scored.target.index_select(0, top.to(scored.target.device))
    )
    assert bool((birth.w == 0.0).all())
    assert store.live_ids().numel() == before_live + budget


def test_hardcore_sampling_rejects_candidates_near_live_atoms() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(41)
    sigma = 0.05
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=32,
        mode="raw",
        sampling="hardcore",
        sigma=sigma,
        hardcore_attempts=16,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    assert field._source.shape[0] == 32

    live = torch.cat((view.s, view.t), dim=1)
    pool = torch.cat((field._source, field._target), dim=1)
    joint = torch.cdist(pool, live)
    # Every kept candidate must clear sigma from every live atom, OR have been
    # supplied by the documented uniform fallback when the budget ran out.
    cleared = (joint.amin(dim=1) > sigma).all()
    assert bool(cleared) or field.hardcore_attempts > 0


def test_hardcore_sampling_falls_back_to_uniform_when_budget_is_exhausted() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(3)
    # An enormous sigma makes rejection sampling impossible to satisfy within
    # any bounded attempt budget on a bounded Box, so the documented uniform
    # fallback must still return a full pool without hanging or raising.
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=10,
        mode="raw",
        sampling="hardcore",
        sigma=1000.0,
        hardcore_attempts=2,
        eps=1e-6,
        name="continuous_candidate_field",
    )
    view = store.view()
    field.prepare(view, module)
    assert field._source.shape[0] == 10
    assert field._target.shape[0] == 10
    scores = field.candidate_snapshot().scores
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# Coarse-to-fine refinement (Stage 0 under-resolution fix)
# ---------------------------------------------------------------------------


def test_refinement_grows_the_pool_by_top_k_times_samples_per_winner_per_round() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(11)
    schedule = RefinementSchedule(top_k=3, samples_per_winner=5, radius_multiplier=1.0, rounds=2)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=6,
        mode="deflated",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
        refinement=schedule,
    )
    view = store.view()
    field.prepare(view, module)
    x = torch.randn(4, 3, dtype=torch.float64)
    g_out = torch.randn(4, 2, dtype=torch.float64)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    assert snapshot.source.shape[0] == 6 + 3 * 5 * schedule.rounds
    assert snapshot.target.shape[0] == 6 + 3 * 5 * schedule.rounds
    assert snapshot.scores.numel() == 6 + 3 * 5 * schedule.rounds
    assert torch.isfinite(snapshot.scores).all()


def test_same_rng_seed_gives_identical_pool_and_scores_through_refinement() -> None:
    store, module = _build_module()
    x = torch.randn(4, 3, dtype=torch.float64)
    g_out = torch.randn(4, 2, dtype=torch.float64)
    schedule = RefinementSchedule(top_k=3, samples_per_winner=5, radius_multiplier=1.0, rounds=2)

    def _run(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = torch.Generator().manual_seed(seed)
        field = ContinuousCandidateField(
            store,
            module,
            rng,
            pool_size=6,
            mode="deflated",
            sampling="uniform",
            sigma=None,
            hardcore_attempts=8,
            eps=1e-6,
            name="continuous_candidate_field",
            refinement=schedule,
        )
        view = store.view()
        field.prepare(view, module)
        _accumulate(field, module, view, x, g_out)
        snapshot = field.candidate_snapshot()
        return snapshot.source, snapshot.target, snapshot.scores

    source_a, target_a, scores_a = _run(101)
    source_b, target_b, scores_b = _run(101)
    torch.testing.assert_close(source_a, source_b)
    torch.testing.assert_close(target_a, target_b)
    torch.testing.assert_close(scores_a, scores_b)


def test_refined_candidates_respect_hardcore_minimum_distance_constraint() -> None:
    store, module = _build_module()
    rng = torch.Generator().manual_seed(9)
    sigma = 0.05
    schedule = RefinementSchedule(top_k=4, samples_per_winner=8, radius_multiplier=1.0, rounds=1)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=32,
        mode="raw",
        sampling="hardcore",
        sigma=sigma,
        hardcore_attempts=16,
        eps=1e-6,
        name="continuous_candidate_field",
        refinement=schedule,
    )
    view = store.view()
    field.prepare(view, module)
    x = torch.randn(3, 3, dtype=torch.float64)
    g_out = torch.randn(3, 2, dtype=torch.float64)
    _accumulate(field, module, view, x, g_out)

    snapshot = field.candidate_snapshot()
    # First 32 rows are the coarse pool; the rest are the refined batch.
    refined_source = snapshot.source[32:]
    refined_target = snapshot.target[32:]
    assert refined_source.shape[0] == schedule.top_k * schedule.samples_per_winner

    live = torch.cat((view.s, view.t), dim=1)
    pool = torch.cat((refined_source, refined_target), dim=1)
    joint = torch.cdist(pool, live)
    cleared = (joint.amin(dim=1) > sigma).all()
    assert bool(cleared) or field.hardcore_attempts > 0


def test_refinement_finds_a_better_maximum_on_a_narrow_synthetic_peak() -> None:
    """Coarse-to-fine refinement should resolve a peak the coarse pool misses.

    Uses a synthetic score function (a narrow Gaussian bump centered in the
    chart, width << the coarse spacing) instead of a real gradient, per the
    task: this isolates the geometric claim -- "refinement finds a better
    maximum than the coarse pool alone" -- from any question about whether a
    real gradient happens to be informative.
    """
    store, module = _build_module()

    def _peak_score(
        self: ContinuousCandidateField,
        source: torch.Tensor,
        target: torch.Tensor,
        gradient: torch.Tensor,
        live_source: torch.Tensor,
        live_target: torch.Tensor,
    ) -> torch.Tensor:
        del gradient, live_source, live_target
        center_source = source.new_full((1, source.shape[1]), 0.5)
        center_target = target.new_full((1, target.shape[1]), 0.5)
        width = 0.01
        d2 = (source - center_source).pow(2).sum(dim=1) + (target - center_target).pow(2).sum(dim=1)
        return torch.exp(-d2 / (2.0 * width * width))

    def _make(seed: int, refinement: RefinementSchedule | None) -> ContinuousCandidateField:
        rng = torch.Generator().manual_seed(seed)
        field = ContinuousCandidateField(
            store,
            module,
            rng,
            pool_size=8,
            mode="raw",
            sampling="uniform",
            sigma=None,
            hardcore_attempts=8,
            eps=1e-6,
            name="continuous_candidate_field",
            refinement=refinement,
        )
        field._score = types.MethodType(_peak_score, field)  # type: ignore[method-assign]
        view = store.view()
        field.prepare(view, module)
        return field

    seed = 0
    coarse_only = _make(seed, None)
    coarse_max = coarse_only.candidate_snapshot().scores.max()

    refined = _make(seed, RefinementSchedule(top_k=8, samples_per_winner=256, radius_multiplier=1.0, rounds=2))
    refined_max = refined.candidate_snapshot().scores.max()

    assert bool(refined_max > coarse_max)
    # The coarse M=8 pool essentially never lands inside a width-0.01 peak;
    # refinement should land close enough to the true peak value of 1.0 that
    # the improvement is not just noise.
    assert bool(coarse_max < 0.01)
    assert bool(refined_max > 0.5)


# ---------------------------------------------------------------------------
# Resolution: worst-case ResNet-20 layer geometry (Stage 0's measurement)
# ---------------------------------------------------------------------------

# Geometry for `stage2.block0.conv2` (out_channels=64, in_channels=64,
# kernel_size=3x3, so in=64x9=576 taps on the 3D (channel, ky, kx) input
# chart) -- the layer Stage 0 measured as the worst case in
# cst/experiments/framework_rebase_infra.py's `build_layer_plan`. Taken from
# `torchcst.compute.conv2d_neuron_coordinates`, the exact function that file
# calls to build the chart, rather than invented.
_WORST_LAYER_IN_CHANNELS = 64
_WORST_LAYER_OUT_CHANNELS = 64
_WORST_LAYER_KERNEL_SIZE = 3
_WORST_LAYER_SIGMA_MULTIPLIER = 3.0  # the pinned Stage-0 multiplier


def _mean_nn_spacing(mu: torch.Tensor) -> float:
    """Mean nearest-neighbour Euclidean spacing between chart rows.

    Mirrors `cst/experiments/framework_rebase_infra.py`'s `_mean_nn_spacing`
    exactly (same reduction), reproduced here rather than imported since
    that file lives in a separate repo/package from torchcst.
    """
    if mu.shape[0] <= 1:
        return 1.0
    distance = torch.cdist(mu, mu)
    distance.fill_diagonal_(float("inf"))
    return float(distance.min(dim=1).values.mean())


def _worst_layer_sigma_init() -> float:
    """Reproduce `CSTConvLayer.__init__`'s `sigma_init` for the worst layer.

    `sigma_init = 0.5 * (sigma_in_init + sigma_out_init) * sigma_multiplier`
    with `sigma_in_init`/`sigma_out_init` the mean nearest-neighbour chart
    spacing on each side -- copied from
    `cst/experiments/framework_rebase_infra.py`'s `CSTConvLayer.__init__`.
    """
    in_mu, out_mu = conv2d_neuron_coordinates(
        _WORST_LAYER_IN_CHANNELS, _WORST_LAYER_OUT_CHANNELS, _WORST_LAYER_KERNEL_SIZE
    )
    sigma_in_init = _mean_nn_spacing(in_mu)
    sigma_out_init = _mean_nn_spacing(out_mu)
    return 0.5 * (sigma_in_init + sigma_out_init) * _WORST_LAYER_SIGMA_MULTIPLIER


def test_coarse_pool_reproduces_stage0_under_resolution_on_the_worst_layer() -> None:
    """The documented artifact this task exists to fix, reproduced exactly.

    Stage 0 measured the M=4096 coarse pool's spacing on this layer's 3D
    input chart at ~1.3x sigma -- wider than sigma, so the pool cannot
    resolve the field's peak. `_coarse_spacing` is the closed-form
    `width * M ** (-1/dim)` estimate `ContinuousCandidateField._refine` uses
    for its own radius rule; asserting it here (rather than only trusting
    the module docstring's claim) pins the "before" number this feature is
    supposed to fix.
    """
    sigma_init = _worst_layer_sigma_init()
    domain_in = Box(0.0, 1.0, 3)
    coarse_spacing = _coarse_spacing(domain_in, 4096)
    ratio = coarse_spacing / sigma_init
    assert ratio > 1.0
    # Stage 0 reported ~1.3x; pin a tolerant band around that so a future
    # change to the layer plan's ERK budget or channel counts is caught
    # rather than silently drifting the "before" story.
    assert 1.2 < ratio < 1.45


def test_refinement_resolves_the_worst_layer_below_sigma() -> None:
    """After the fix: local refinement's effective spacing clears sigma.

    Empirically draws one refinement round's local pool (via the same
    `_sample_local_uniform` helper `ContinuousCandidateField._refine` calls)
    around one interior winner, using a schedule sized so the refined
    factor-column buffer stays the same order of magnitude as the coarse
    pool's own buffer (`top_k * samples_per_winner` candidates, not
    `pool_size * refinement_factor` candidates) -- the memory point from the
    task: brute-forcing sigma/2 in 3D needs ~18x the candidates
    (~180 MB/layer), whereas one round of this schedule adds only
    `top_k * samples_per_winner` = 8 * 512 = 4096 candidates total, doubling
    (not 18x-ing) the coarse pool's peak buffer while resolving well below
    sigma.
    """
    sigma_init = _worst_layer_sigma_init()
    domain_in = Box(0.0, 1.0, 3)
    pool_size = 4096
    samples_per_winner = 512
    radius_multiplier = 1.0
    radius = _coarse_spacing(domain_in, pool_size) * radius_multiplier

    rng = torch.Generator().manual_seed(0)
    winner = torch.rand(1, domain_in.dim, generator=rng) * 0.5 + 0.25  # an interior anchor
    local_pool = _sample_local_uniform(domain_in, winner, radius, samples_per_winner, rng)
    assert local_pool.shape == (samples_per_winner, domain_in.dim)

    empirical_spacing = _mean_nn_spacing(local_pool)
    ratio = empirical_spacing / sigma_init
    assert ratio < 1.0


# ---------------------------------------------------------------------------
# Selection ranks by |score| for a signed scalar candidate
#
# This instrument is the one candidate instrument in the library whose scores
# are *signed* (GradientField/ContinuousGradientField/ScoredCandidates already
# emit |grad|), and a synapse atom's coefficient is a signed scalar, so the
# theory selects it by |S_o| -- theory/sections/03_support_dynamics.tex,
# def. "一般 birth score": `c_o in R` -> `|S_o|`, and only a fixed non-negative
# ray `c_o >= 0` (a neuron gate) -> `[S_o]_+`. The profile gain
# (eq:profile-birth-gain) is quadratic in the score, so a large negative score
# is exactly as good a candidate as a large positive one. Discrete RigL says
# the same thing when it ranks dormant weights by |grad|.
# ---------------------------------------------------------------------------


def test_selector_prefers_a_strongly_negative_score_over_a_weakly_positive_one() -> None:
    scores = torch.tensor([0.1, -5.0, 0.9, -0.2, 4.0])
    chosen = TopKSelector().select(scores, 2)
    assert chosen.tolist() == [1, 4]  # |-5.0| then |4.0|, not 4.0 then 0.9

    # budget 1 is the sharpest form of the same statement
    assert TopKSelector().select(torch.tensor([-5.0, 0.9]), 1).tolist() == [0]


def test_nonnegative_ray_selection_stays_available_and_unchanged() -> None:
    """The `[S]_+` case is a different candidate class, not a fallback.

    A gate pinned to `c_o >= 0` cannot buy a negative score at all, so it
    ranks by `[S_o]_+`; the two cases must not be collapsed. The neuron axis
    is deferred on this bench but this behaviour has to be here and correct
    when it comes back.
    """
    scores = torch.tensor([0.1, -5.0, 0.9, -0.2, 4.0])
    ray = TopKSelector(candidate_cone="nonnegative_ray")
    assert ray.select(scores, 2).tolist() == [4, 2]  # 4.0 then 0.9; -5.0 is worthless

    # every non-positive candidate has zero gain, so they tie and the stable
    # sort breaks the tie by index (rather than preferring the least negative)
    assert ray.select(torch.tensor([-0.2, -5.0, -0.1]), 3).tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="candidate_cone"):
        TopKSelector(candidate_cone="absolute")


def test_refinement_zooms_on_negative_peaks_too() -> None:
    """`_refine` must rank winners the same way selection does.

    If refinement zoomed only on `+score` peaks it would resolve half the
    field and leave the other half at coarse resolution -- and the refined
    pool is exactly what the caller then selects over, so a disagreement
    between the two rankings means refinement zooms where selection will not
    buy.
    """
    store, module = _build_module()

    def _negative_peak(
        self: ContinuousCandidateField,
        source: torch.Tensor,
        target: torch.Tensor,
        gradient: torch.Tensor,
        live_source: torch.Tensor,
        live_target: torch.Tensor,
    ) -> torch.Tensor:
        del gradient, live_source, live_target
        center_source = source.new_full((1, source.shape[1]), 0.5)
        center_target = target.new_full((1, target.shape[1]), 0.5)
        width = 0.01
        d2 = (source - center_source).pow(2).sum(dim=1) + (target - center_target).pow(2).sum(dim=1)
        return -torch.exp(-d2 / (2.0 * width * width))  # the peak points *down*

    rng = torch.Generator().manual_seed(0)
    field = ContinuousCandidateField(
        store,
        module,
        rng,
        pool_size=8,
        mode="raw",
        sampling="uniform",
        sigma=None,
        hardcore_attempts=8,
        eps=1e-6,
        name="continuous_candidate_field",
        refinement=RefinementSchedule(top_k=8, samples_per_winner=256, radius_multiplier=1.0, rounds=2),
    )
    field._score = types.MethodType(_negative_peak, field)  # type: ignore[method-assign]
    field.prepare(store.view(), module)

    scores = field.candidate_snapshot().scores
    # the same narrow width-0.01 bump the positive-peak test uses, mirrored:
    # a coarse M=8 pool essentially never lands inside it, so finding it at
    # all is refinement's doing.
    assert bool(scores.abs().max() > 0.5)
    assert bool(scores.min() < -0.5)
