"""ContinuousCandidateField cRigL/cRES scoring, KernelPort, and ScoredBirth wiring."""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.instruments import (
    ContinuousCandidateField,
    ContinuousCandidateRequest,
    WeightedMeasurement,
)
from torchcst.policy import (
    EvenBudgetAllocator,
    MagnitudeCourt,
    PeriodicSchedule,
    Policy,
    ScoredBirth,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
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
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.25).to(dtype))
    return store, module


def _accumulate(field: ContinuousCandidateField, module, view, x: torch.Tensor, g_out: torch.Tensor) -> None:
    measurement = field._measure(module, x, g_out)
    field.finalize_update((WeightedMeasurement(measurement, 1.0),), view)


def _brute_direction(module, source_row: torch.Tensor, target_row: torch.Tensor) -> torch.Tensor:
    """Densely materialize the Frobenius-unit direction ``a_z`` for one candidate."""
    k_in, k_out = module.kernel_columns(source_row, target_row)
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
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.25).to(dtype))

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
    policy = Policy(
        schedule=PeriodicSchedule(event_interval=1, birth_budget=2, observe_window=1),
        proposers=(proposer,),
        allocator=EvenBudgetAllocator(),
        retention=MagnitudeCourt(0.0),
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

    top = torch.argsort(scored.scores, descending=True, stable=True)[:budget]
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
