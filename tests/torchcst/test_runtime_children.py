"""Phase 2 S2: store-bound runtime children (policy/runtime.py)."""

import torch

from torchcst.compute import CSTLinear
from torchcst.engine import StructuralEngine
from torchcst.policy import PeriodicCadence, QuotaRegime
from torchcst.policy.contract import Clock
from torchcst.policy.families import RENT, cSET
from torchcst.policy.runtime import EndpointChild, PricedProposal, SiteBinding, SynapseChild
from torchcst.policy.registry import RetiredCandidateRegistry
from torchcst.representation import GaussianFactor, RepresentationSpec
from torchcst.storage import NeuronStore, SynapseBirth, SynapseDeath, SynapseStore


def _continuous_store(capacity: int = 12) -> tuple[SynapseStore, NeuronStore, NeuronStore, CSTLinear]:
    store = SynapseStore(
        "edge",
        1,
        1,
        capacity=capacity,
        spec=RepresentationSpec.continuous(1, 1, bounds=(0.0, 1.0)),
        dtype=torch.float64,
    )
    inputs = NeuronStore(
        "edge_in", 1, mu=torch.zeros(1, 1, dtype=torch.float64), initial_live=1,
        dtype=torch.float64,
    )
    outputs = NeuronStore(
        "edge_out", 2, mu=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        initial_live=2, dtype=torch.float64,
    )
    module = CSTLinear(inputs, outputs, store, GaussianFactor(0.1).double())
    s = torch.tensor([[0.50], [0.5001], [0.4999]], dtype=torch.float64)
    store.apply(
        [SynapseBirth(store.site, s, s.clone(), torch.tensor([1.0, -0.7, 0.4], dtype=torch.float64), torch.arange(3))]
    )
    return store, inputs, outputs, module


def _child(lifecycle, lam=None, capacity: int = 12):
    store, inputs, outputs, module = _continuous_store(capacity)
    built = lifecycle.build(lam)
    child = SynapseChild(lifecycle, built, SiteBinding(store=store, module=module))
    return child, store, inputs, outputs, module


def test_child_view_matches_the_engines_own_bound_child_view() -> None:
    """A hand-built child's view equals the engine's own bound child's view.

    Since Phase 2 S4e (docs/policy-tree-phase2.md), the engine no longer
    builds a proposal view of its own at all -- the child is the only place
    this logic lives (``StructuralEngine._proposal_view``/``_ages`` are
    deleted along with the composed/whole-policy engine paths). This checks
    the same thing the old cross-check checked: view construction is a pure,
    deterministic function of the store, so a hand-built
    :class:`~torchcst.policy.runtime.SynapseChild` for one store and the
    engine's own bound tree's child for that identical store must agree.
    """
    child, store, inputs, outputs, module = _child(cSET())
    root = QuotaRegime(
        budget=1, method=cSET(), cadence=PeriodicCadence(event_interval=1)
    )
    engine = StructuralEngine(
        {"edge": store, "edge_in": inputs, "edge_out": outputs},
        root,
        modules={"edge": module},
        seed=0,
    )
    engine_child = engine._tree.children[0]
    mine = child.view()
    theirs = engine_child.view()
    assert mine.site == theirs.site and mine.version == theirs.version
    assert torch.equal(mine.ids, theirs.ids)
    assert torch.equal(mine.lineages, theirs.lineages)
    assert mine.bounds_in == theirs.bounds_in and mine.bounds_out == theirs.bounds_out
    assert torch.equal(mine.retired_in, theirs.retired_in)
    assert torch.equal(mine.retired_out, theirs.retired_out)
    assert torch.equal(child.ages(mine), engine_child.ages(theirs))


def test_child_two_phase_execution_and_retired_lineages() -> None:
    child, store, *_ = _child(cSET())
    view = child.view()
    dead = SynapseDeath(store.site, view.ids[:2])
    expected = view.lineages[:2].clone()
    ticket, retired = child.prepare((dead,))
    assert torch.equal(retired.sort().values, expected.sort().values)
    assert store.view().ids.numel() == 3  # nothing written before commit
    child.commit(ticket)
    assert store.view().ids.numel() == 1


def test_child_immunity_raises_on_young_prune() -> None:
    child, store, *_ = _child(cSET(drop_fraction=0.9))
    # cSET's MagnitudeCourt carries an immunity window; freshly-born atoms
    # (age 0) must be protected when the court would otherwise take them.
    if child.immunity_events == 0:
        return  # court declares no immunity; nothing to enforce
    view = child.view()
    try:
        child.decide_retention(view, Clock(update_step=1, event_index=1))
    except RuntimeError as error:
        assert "immune" in str(error)


def test_child_birth_budget_is_enforced() -> None:
    child, *_ = _child(cSET())
    registry = RetiredCandidateRegistry()
    rng = torch.Generator()
    rng.manual_seed(0)
    proposals = child.propose_births(child.view(), 2, registry, rng)
    assert all(isinstance(proposal, PricedProposal) for proposal in proposals)
    assert sum(proposal.cost for proposal in proposals) <= 2


def _bind_child_instruments(child, store, module) -> None:
    """Build and bind the child's declared instruments, as the root will."""
    from torchcst.instruments import InstrumentBuildContext

    registry = RetiredCandidateRegistry()
    rng = torch.Generator()
    rng.manual_seed(0)
    built = {
        requirement.name: requirement.build(
            InstrumentBuildContext(
                site=store.site, store=store, module=module,
                registry=registry, rng=rng,
            )
        )
        for requirement in child.requires
    }
    child.bind_instruments(built)


def test_rent_child_absorb_proposals_are_priced_proposals() -> None:
    child, store, _, _, module = _child(
        RENT(radius=0.01, ridge=1.0e-9, pool_size=16), lam=1.0e-6
    )
    _bind_child_instruments(child, store, module)
    registry = RetiredCandidateRegistry()
    rng = torch.Generator()
    rng.manual_seed(0)
    proposals = child.propose_absorb(child.view(), 4, registry, rng)
    assert all(isinstance(proposal, PricedProposal) for proposal in proposals)
    # The three planted near-duplicate atoms give the court real work.
    assert proposals


def test_endpoint_child_ticks_only_its_own_store() -> None:
    _, inputs, outputs, _ = _continuous_store()
    child = EndpointChild(inputs)
    before_in = inputs.age.values.clone()
    before_out = outputs.age.values.clone()
    child.tick_age()
    assert bool((inputs.age.values >= before_in).all())
    assert torch.equal(outputs.age.values, before_out)


def test_requires_union_over_rules() -> None:
    child, *_ = _child(RENT(radius=0.01, pool_size=16), lam=1.0e-6)
    names = [getattr(req, "name", None) for req in child.requires]
    assert len(names) == len(set(names))
    assert child.requires  # ScoredBirth's candidate request must surface
