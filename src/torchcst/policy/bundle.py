"""Atomic cross-store proposal bundles and response composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torchcst.storage import (
    NeuronKick,
    NeuronRetire,
    NeuronStore,
    NeuronUngate,
    SynapseBirth,
    SynapseDeath,
    SynapseKick,
    SynapseMerge,
)


Op = (
    SynapseBirth
    | SynapseDeath
    | SynapseMerge
    | SynapseKick
    | NeuronUngate
    | NeuronRetire
    | NeuronKick
)


@dataclass(frozen=True)
class ProposalBundle:
    """One indivisible proposal; operations may target different sites."""

    bundle_id: str
    ops: tuple[Op, ...]
    atomic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_id, str) or not self.bundle_id:
            raise ValueError("bundle_id must be a non-empty string")
        if not isinstance(self.ops, tuple):
            raise TypeError("ops must be a tuple")
        if not isinstance(self.atomic, bool):
            raise TypeError("atomic must be a bool")


def bundle_birth_count(bundle: ProposalBundle) -> int:
    """Return the schedule-priced synapse births carried by one bundle."""
    if not isinstance(bundle, ProposalBundle):
        raise TypeError("bundle must be a ProposalBundle")
    return sum(
        int(op.w.numel()) for op in bundle.ops if isinstance(op, SynapseBirth)
    )


@dataclass(frozen=True)
class BundleComposer:
    """Declaratively join one scheduled ungate with its incident birth block."""

    incident_births: int = 1
    initial_gate: float = NeuronStore.DEFAULT_UNGATE
    bundle_prefix: str = "response"

    def __post_init__(self) -> None:
        if isinstance(self.incident_births, bool) or not isinstance(
            self.incident_births, int
        ):
            raise TypeError("incident_births must be an int")
        if self.incident_births <= 0:
            raise ValueError("incident_births must be positive")
        if not torch.isfinite(torch.tensor(float(self.initial_gate))):
            raise ValueError("initial_gate must be finite")
        if not isinstance(self.bundle_prefix, str) or not self.bundle_prefix:
            raise ValueError("bundle_prefix must be a non-empty string")

    @staticmethod
    def compose(
        bundle_id: str, ops: tuple[Op, ...], atomic: bool = True
    ) -> ProposalBundle:
        """Construct a general bundle without interpreting its operations."""
        return ProposalBundle(bundle_id, ops, atomic)

    def compose_response(
        self,
        *,
        event_index: int,
        neuron_store: NeuronStore,
        synapse_view: Any,
        proposer: Any,
        registry: Any,
        rng: torch.Generator,
        neuron_id: int | None = None,
        birth_budget: int | None = None,
    ) -> ProposalBundle | None:
        """Build ``ungate + G output-incident births`` for the next dormant ID.

        A short incident block is not emitted: activating a neuron without the
        declared number of incident atoms would recreate the half-equilibrium
        ProposalBundle is designed to exclude.
        """
        dormant = neuron_store.dormant_ids()
        if dormant.numel() == 0:
            return None
        target = int(dormant[0]) if neuron_id is None else int(neuron_id)
        if target not in dormant.tolist():
            raise ValueError("response target must be DORMANT")
        count = self.incident_births
        if birth_budget is not None:
            count = min(count, int(birth_budget))
        if count < self.incident_births:
            return None
        proposed = tuple(
            proposer.propose_incident(
                synapse_view, self.incident_births, target, registry, rng
            )
        )
        if sum(
            int(op.w.numel()) for op in proposed if isinstance(op, SynapseBirth)
        ) != self.incident_births:
            return None
        bundle_id = (
            f"{self.bundle_prefix}:{event_index}:{neuron_store.site}:{target}"
        )
        return ProposalBundle(
            bundle_id,
            (
                NeuronUngate(
                    neuron_store.site,
                    torch.tensor([target], dtype=torch.int64),
                    self.initial_gate,
                ),
                *proposed,
            ),
        )
