"""Declarative action rules replace planner and legacy component fields."""

from __future__ import annotations

import pytest
import torch

from torchcst.engine import StructuralEngine
from torchcst.policy import (
    ActionKind,
    ActionSpec,
    ConstantQuota,
    EvenBudgetDistributor,
    MergeProposer,
    PeriodicCadence,
    Policy,
    StructuralQuota,
)
from torchcst.representation import RepresentationSpec
from torchcst.storage import SynapseBirth, SynapseMerge, SynapseStore


def test_merge_action_consumes_merge_quota_without_birth_or_prune_actions() -> None:
    store = SynapseStore(
        "rank",
        2,
        2,
        capacity=2,
        spec=RepresentationSpec.rank_one(2, 2),
    )
    store.apply(
        [
            SynapseBirth(
                store.site,
                torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
                torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
                torch.tensor([0.5, 0.25]),
                torch.tensor([0, 1], dtype=torch.int64),
            )
        ]
    )
    policy = Policy(
        cadence=PeriodicCadence(event_interval=1),
        quota=ConstantQuota(StructuralQuota(synapse_merge=1)),
        actions=(ActionSpec.synapse_merge(MergeProposer(0.9)),),
        distributor=EvenBudgetDistributor(),
    )

    operations = StructuralEngine({store.site: store}, policy).step()
    assert any(isinstance(operation, SynapseMerge) for operation in operations)
    assert store.live_ids().numel() == 1
    assert policy.proposal_actions[0].kind is ActionKind.SYNAPSE_MERGE


def test_policy_rejects_mixed_new_and_legacy_action_surfaces() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        Policy(
            cadence=PeriodicCadence(event_interval=1),
            quota=ConstantQuota(StructuralQuota()),
            actions=(ActionSpec.synapse_merge(MergeProposer()),),
            proposers=(MergeProposer(),),
            distributor=EvenBudgetDistributor(),
        )
