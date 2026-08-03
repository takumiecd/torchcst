"""cSFW's periodic backfit: the shipped default, position polish included.

Every other fast-construction test disables the backfit (``backfit=None``) to
isolate pure growth, which left ``cSFW()``'s own defaults -- a ``K/10``
cadence with one damped-Newton position iteration -- unexercised.  The
backfit is what makes ``polish_iters=0`` a *deferred* payment rather than no
payment at all (FC-0's ruling 1), so it belongs in the contract.
"""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import (
    EvenBudgetDistributor,
    PeriodicCadence,
    QuotaRegime,
    cSFW,
    cVP,
)
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import (
    NeuronStore,
    SynapseBirth,
    SynapseDeath,
    SynapseRefit,
    SynapseStore,
)


def _parts(
    method,
    *,
    seed: int = 4,
    source: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    store = SynapseStore(
        "fc",
        1,
        1,
        16,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    if source is None:
        source = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    if target is None:
        target = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    if weight is None:
        weight = torch.tensor([0.05, -0.05], dtype=torch.float64)
    store.apply(
        [
            SynapseBirth(
                store.site,
                source,
                target,
                weight,
                torch.arange(weight.numel()),
            )
        ]
    )
    mu = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    inputs = NeuronStore("fc_in", 3, mu=mu, initial_live=3, dtype=torch.float64)
    outputs = NeuronStore("fc_out", 3, mu=mu, initial_live=3, dtype=torch.float64)
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.3).double())
    root = QuotaRegime(
        budget=1,
        method=method,
        cadence=PeriodicCadence(event_interval=1, observe_window=1),
        distributor=EvenBudgetDistributor(),
    )
    engine = StructuralEngine(
        {store.site: store, inputs.site: inputs, outputs.site: outputs},
        root,
        modules={store.site: module},
        seed=seed,
    )
    return engine, module, store


def _problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(32, 3, dtype=torch.float64, generator=generator)
    dense = torch.tensor(
        [[1.0, -0.5, 0.2], [0.3, 0.8, -1.1], [-0.4, 0.1, 0.9]],
        dtype=torch.float64,
    )
    return x, x @ dense.t()


def _run_event(
    engine: StructuralEngine,
    module: CSTLinear,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[object, ...]:
    module.zero_grad(set_to_none=True)
    engine.begin_update()
    (module(x) - y).square().mean().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    return engine.step()


def _loss(module: CSTLinear, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((module(x) - y).square().mean())


def test_default_backfit_solves_the_whole_table_and_moves_positions() -> None:
    engine, module, store = _parts(cSFW(pool_size=64, multistart=2))
    x, y = _problem()
    before = _loss(module, x, y)

    per_event = [_run_event(engine, module, x, y) for _ in range(4)]

    for ops in per_event:
        refits = [op for op in ops if isinstance(op, SynapseRefit)]
        assert len(refits) == 1
        refit = refits[0]
        # The default backfit pays for both halves: every live amplitude is
        # re-solved, and one damped-Newton iteration moves the coordinates.
        assert refit.s is not None and refit.t is not None
        assert torch.equal(refit.ids.sort().values, refit.ids)
        # Refits land before births in the event order, so the table addresses
        # the atoms that were live when the evidence was measured.
        assert [type(op) for op in ops] == [SynapseRefit, SynapseBirth]
    assert not any(isinstance(op, SynapseDeath) for ops in per_event for op in ops)
    # The trust region and the box clamp keep polished coordinates in-domain.
    assert bool((store.s >= 0.0).all() and (store.s <= 1.0).all())
    assert bool((store.t >= 0.0).all() and (store.t <= 1.0).all())
    assert _loss(module, x, y) < before


def test_backfit_period_counts_births_not_events() -> None:
    engine, module, store = _parts(cSFW(backfit=3, pool_size=64, multistart=2))
    x, y = _problem()

    trace: list[tuple[int, bool]] = []
    for _ in range(7):
        ops = _run_event(engine, module, x, y)
        trace.append(
            (
                int(store.live_ids().numel()),
                any(isinstance(op, SynapseRefit) for op in ops),
            )
        )

    refit_at = [count for count, refit in trace if refit]
    assert refit_at, "a period-3 backfit must fire at least once in 7 births"
    assert len(refit_at) < len(trace), "period 3 must be sparser than every event"
    # Consecutive backfits are at least a period apart in live-atom count.
    assert all(
        later - earlier >= 3 for earlier, later in zip(refit_at, refit_at[1:])
    )


def test_backfit_none_never_emits_a_refit() -> None:
    engine, module, _ = _parts(cSFW(backfit=None, pool_size=64, multistart=2))
    x, y = _problem()

    per_event = [_run_event(engine, module, x, y) for _ in range(4)]

    assert not any(isinstance(op, SynapseRefit) for ops in per_event for op in ops)
    assert any(isinstance(op, SynapseBirth) for ops in per_event for op in ops)


def test_backfit_clamps_a_twin_gram_explosion_to_the_pre_refit_live_scale() -> None:
    # Two near-duplicate atoms (twins, twin-control.md Sec.1) make the event
    # -time tangent Gram near-singular. A tiny ridge lets the raw solve
    # return a huge cancelling pair -- the FC-5 diagnostic saw a live w2max
    # jump from 6.48 to 5.33e27 this way. TangentBirth already bounds every
    # solved amplitude at the pre-event live scale; TangentRefit must do the
    # same for its *applied* weights (not the delta).
    gap = 1.0e-8
    engine, module, store = _parts(
        cVP(ridge=1.0e-12),
        source=torch.tensor([[0.5], [0.5 + gap]], dtype=torch.float64),
        target=torch.tensor([[0.5], [0.5 + gap]], dtype=torch.float64),
        weight=torch.tensor([0.05, -0.05], dtype=torch.float64),
    )
    x, y = _problem()
    live_scale_before = float(store.view().w.detach().abs().max())

    ops = _run_event(engine, module, x, y)

    refits = [op for op in ops if isinstance(op, SynapseRefit)]
    assert len(refits) == 1
    refit = refits[0]
    assert bool(torch.isfinite(refit.w).all())
    assert float(refit.w.abs().max()) <= live_scale_before + 1.0e-9


def test_backfit_clamp_is_a_no_op_away_from_any_twin() -> None:
    # Two well-separated atoms with a live scale large enough that the
    # solved correction never approaches it: the clamp added alongside the
    # twin case above must be transparent here, i.e. behave exactly as
    # backfit did before that fix. Verified two ways: the applied weights
    # land strictly inside the clamp bounds (so ``clamp`` was a no-op), and
    # they match the value backfit produced before the clamp existed.
    weight = torch.tensor([1.0, -1.0], dtype=torch.float64)
    engine, module, store = _parts(cVP(ridge=1.0e-4), weight=weight)
    x, y = _problem()
    live_scale_before = float(store.view().w.detach().abs().max())

    ops = _run_event(engine, module, x, y)

    refits = [op for op in ops if isinstance(op, SynapseRefit)]
    assert len(refits) == 1
    refit = refits[0]
    assert bool((refit.w.abs() < live_scale_before).all())
    torch.testing.assert_close(
        refit.w,
        torch.tensor([0.8239584471648637, -0.22970732684406459], dtype=torch.float64),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
