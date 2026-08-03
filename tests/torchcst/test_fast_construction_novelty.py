"""cSFW's opt-in gain_perp novelty discount (twin-control.md Sec.3, rung 1).

``TangentBirth.novelty`` -- exposed via ``cSFW(novelty=sigma)`` -- discounts
each candidate's gain by its worst-case coordinate overlap with live atoms
and atoms already accepted earlier in the same birth event, so a candidate
that is a near twin of something already present is priced down toward zero
rather than born. ``novelty=None`` (the default) must reproduce the
pre-existing, undiscounted selection exactly.
"""

from __future__ import annotations

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import EvenBudgetDistributor, PeriodicCadence, QuotaRegime, cSFW
from torchcst.representation import GaussianKernel, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseStore


def _twin_parts(
    *, novelty: float | None, seed: int = 4
) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    """One live atom at (0.5, 0.5); the residual's true peak sits on top of it.

    The target is exactly twice what the live atom already produces, so the
    residual after subtracting the live fit has the identical functional
    shape as the live atom itself: the unconstrained best correction is a
    twin of the atom that is already live.
    """
    store = SynapseStore(
        "fc", 1, 1, 16,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.5]], dtype=torch.float64),
                torch.tensor([[0.5]], dtype=torch.float64),
                torch.tensor([0.3], dtype=torch.float64),
                torch.arange(1),
            )
        ]
    )
    mu = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    inputs = NeuronStore("fc_in", 3, mu=mu, initial_live=3, dtype=torch.float64)
    outputs = NeuronStore("fc_out", 3, mu=mu, initial_live=3, dtype=torch.float64)
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.1).double())
    root = QuotaRegime(
        budget=1,
        method=cSFW(
            backfit=None, pool_size=4096, multistart=4, novelty=novelty
        ),
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


def _run_twin_event(engine, module, x):
    with torch.no_grad():
        y = 2.0 * module(x)
    module.zero_grad(set_to_none=True)
    engine.begin_update()
    (module(x) - y).square().mean().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    return engine.step()


def test_novelty_none_reproduces_the_undiscounted_twin_birth() -> None:
    engine, module, _ = _twin_parts(novelty=None)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(64, 3, dtype=torch.float64, generator=generator)

    ops = _run_twin_event(engine, module, x)

    births = [op for op in ops if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    birth = births[0]
    live = torch.tensor([0.5, 0.5], dtype=torch.float64)
    landed = torch.cat((birth.s[0], birth.t[0]))
    # Undiscounted: the true residual peak is a twin of the live atom, and
    # that is exactly what gets born (up to pool-sampling resolution).
    assert float(torch.linalg.vector_norm(landed - live)) < 0.02


def test_novelty_discounts_the_twin_away_from_the_live_atom() -> None:
    engine, module, _ = _twin_parts(novelty=0.1)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(64, 3, dtype=torch.float64, generator=generator)

    ops = _run_twin_event(engine, module, x)

    births = [op for op in ops if isinstance(op, SynapseBirth)]
    assert len(births) == 1
    birth = births[0]
    live = torch.tensor([0.5, 0.5], dtype=torch.float64)
    landed = torch.cat((birth.s[0], birth.t[0]))
    # Discounted: the candidate that used to win (a twin of the live atom)
    # is now priced toward zero gain, so birth lands measurably elsewhere.
    assert float(torch.linalg.vector_norm(landed - live)) > 0.15


def _far_parts(
    *, novelty: float | None, seed: int = 4
) -> tuple[StructuralEngine, CSTLinear, SynapseStore]:
    store = SynapseStore(
        "fc", 1, 1, 16,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[0.25], [0.75]], dtype=torch.float64),
                torch.tensor([[0.25], [0.75]], dtype=torch.float64),
                torch.tensor([0.05, -0.05], dtype=torch.float64),
                torch.arange(2),
            )
        ]
    )
    mu = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    inputs = NeuronStore("fc_in", 3, mu=mu, initial_live=3, dtype=torch.float64)
    outputs = NeuronStore("fc_out", 3, mu=mu, initial_live=3, dtype=torch.float64)
    module = CSTLinear(inputs, outputs, store, GaussianKernel(0.3).double())
    root = QuotaRegime(
        budget=1,
        method=cSFW(
            backfit=None, pool_size=4096, multistart=4, novelty=novelty
        ),
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


def _far_problem() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(32, 3, dtype=torch.float64, generator=generator)
    dense = torch.tensor(
        [[1.0, -0.5, 0.2], [0.3, 0.8, -1.1], [-0.4, 0.1, 0.9]],
        dtype=torch.float64,
    )
    return x, x @ dense.t()


def _run_far_event(engine, module, x, y):
    module.zero_grad(set_to_none=True)
    engine.begin_update()
    (module(x) - y).square().mean().backward()
    engine.observe_microbatch()
    engine.finalize_backward()
    return engine.step()


def test_novelty_leaves_a_far_candidate_unaffected() -> None:
    # Live atoms sit at (0.25, 0.25) and (0.75, 0.75); the true best
    # correction for this unrelated target lands nowhere near either one, so
    # the novelty discount there is negligible and must not move the pick.
    x, y = _far_problem()

    engine, module, _ = _far_parts(novelty=None)
    baseline = [op for op in _run_far_event(engine, module, x, y) if isinstance(op, SynapseBirth)][0]

    engine, module, _ = _far_parts(novelty=0.05)
    discounted = [op for op in _run_far_event(engine, module, x, y) if isinstance(op, SynapseBirth)][0]

    torch.testing.assert_close(baseline.s, discounted.s)
    torch.testing.assert_close(baseline.t, discounted.t)
    torch.testing.assert_close(baseline.w, discounted.w)
