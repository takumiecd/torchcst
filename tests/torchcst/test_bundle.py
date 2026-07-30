"""ProposalBundle accounting contracts.

Phase 2 S4e retires ``StructuralEngine.apply_proposals`` (docs/policy-tree-
phase2.md "消すもの"): the engine no longer accepts a caller-supplied list of
raw ops/bundles to apply out-of-band -- every op now originates from the
bound tree's own ``propose()`` (``policy/tree.py``'s ``RuntimeTree``), and
two-phase commit is an internal root-coordinated affair
(``RuntimeTree._two_phase``), not a public entry point. Three tests that
injected synthetic bundles through ``engine.apply_proposals(...)`` are
retired along with it:

- ``test_invalid_bundle_drops_only_it_and_never_applies_its_ungate`` verified
  the old *partial*-apply semantics (one invalid bundle is dropped, other
  proposals in the same call still commit). That is no longer the behavior
  at all: ruling 1 (docs/policy-tree-phase2.md) makes a prepare failure
  abort the *whole* event, nothing partial -- the direct successor is
  ``test_tree_runtime.py::test_prepare_failure_aborts_the_whole_event``.
- ``test_entry_retirement_cascades_both_endpoint_sides_in_one_plan`` verified
  the retire -> incident-synapse-death cascade by injecting a raw
  ``NeuronRetire`` directly. That cascade is now planned entirely inside
  ``RuntimeTree.propose`` from a real neuron court's own decision; the same
  cross-store cascade property is exercised end-to-end (through a real
  ``NeuronLifecycle`` retention court, not an injected op) by
  ``test_lc_response.py::test_scripted_response_bundle_immunity_rent_cascade_and_requiescence``.
- ``test_continuous_retirement_is_gate_only_until_projection_op_exists``
  verified the continuous/rank-one families' gate-only (no cascade)
  retirement the same injected way; its direct successor is
  ``test_tree_runtime.py::test_continuous_family_neuron_retirement_is_gate_only``.
"""

from __future__ import annotations

import torch

from torchcst.policy import EvenBudgetDistributor, ProposalBundle
from torchcst.storage import NeuronUngate, SynapseBirth


def birth(site: str, count: int, *, target: int = 0) -> SynapseBirth:
    return SynapseBirth(
        site,
        torch.arange(count, dtype=torch.int64).reshape(count, 1),
        torch.full((count, 1), target, dtype=torch.int64),
        torch.ones(count),
        torch.arange(count, dtype=torch.int64),
    )


def test_distributor_accounts_birth_rows_and_never_splits_a_bundle() -> None:
    first = ProposalBundle(
        "first", (NeuronUngate("n", torch.tensor([0])), birth("s", 2))
    )
    second = ProposalBundle(
        "second", (NeuronUngate("n", torch.tensor([1])), birth("s", 2))
    )

    accepted = EvenBudgetDistributor().allocate_bundles(3, (first, second))

    assert accepted == (first,)
    assert accepted[0].ops == first.ops
