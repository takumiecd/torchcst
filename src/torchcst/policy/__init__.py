"""Public policy surface: the tree-native linear stack (docs/policy-tree-phase2.md).

``engine -> root -> children -> storage``, one direction only. A root
(:class:`RentEconomy` or :class:`QuotaRegime`) is built from named methods
(:func:`cSET`, :func:`cRigL`, :func:`cRES`, :func:`RENT`) or a
hand-composed :class:`~torchcst.policy.families.SynapseLifecycle`, and
``root.bind(...)`` produces the live :class:`~torchcst.policy.tree.RuntimeTree`
:class:`~torchcst.engine.StructuralEngine` drives. :mod:`recipes` holds the
few named, validated whole-tree assemblies (``LC``, ``LC_response``,
``GrowthByProfit``).
"""

from .absorb import AbsorbAuditEntry, AbsorbCourt
from .bundle import BundleComposer, Op, ProposalBundle, bundle_birth_count
from .contract import (
    BudgetDistributor,
    BudgetRequest,
    Cadence,
    Clock,
    EvenBudgetDistributor,
    EventSignal,
    InstrumentSpec,
    ObservationRequest,
    OpProposer,
    Phase,
    RetentionCourt,
    QuotaPolicy,
    StructuralQuota,
)
from .courts import MagnitudeCourt, RentCourt
from .proposers import (
    GradFieldTopKBirth,
    IncidentOutputBirth,
    MergeProposer,
    OrthogonalBirth,
    UniformBirth,
    UniformEntryBirth,
)
from .profit import ProfitCourt, TrialSession, TrialTransaction
from .registry import RetiredCandidateRegistry
from .runtime import (
    EndpointChild,
    InterfaceChild,
    PricedProposal,
    SiteBinding,
    SynapseChild,
)
from .scored import ScoredBirth, TopKSelector
from .cadences import BirthWindowCadence, PeriodicCadence
from .quotas import CallableQuota, ConstantQuota, QuotaWindow, WindowedQuota
from .families import NeuronLifecycle, RENT, SynapseLifecycle, cRES, cRigL, cSET, thinned
from .tree import QuotaRegime, RentEconomy, RuntimeTree
from . import recipes

__all__ = [
    "AbsorbAuditEntry",
    "AbsorbCourt",
    "BirthWindowCadence",
    "BudgetDistributor",
    "BudgetRequest",
    "bundle_birth_count",
    "BundleComposer",
    "Clock",
    "Cadence",
    "CallableQuota",
    "ConstantQuota",
    "EndpointChild",
    "EvenBudgetDistributor",
    "EventSignal",
    "GradFieldTopKBirth",
    "InstrumentSpec",
    "InterfaceChild",
    "ObservationRequest",
    "IncidentOutputBirth",
    "MagnitudeCourt",
    "MergeProposer",
    "NeuronLifecycle",
    "OrthogonalBirth",
    "Op",
    "OpProposer",
    "PeriodicCadence",
    "Phase",
    "PricedProposal",
    "ProposalBundle",
    "ProfitCourt",
    "QuotaRegime",
    "RentCourt",
    "RentEconomy",
    "RENT",
    "QuotaPolicy",
    "QuotaWindow",
    "recipes",
    "RetentionCourt",
    "RetiredCandidateRegistry",
    "RuntimeTree",
    "ScoredBirth",
    "SiteBinding",
    "StructuralQuota",
    "SynapseChild",
    "SynapseLifecycle",
    "thinned",
    "TrialSession",
    "TrialTransaction",
    "TopKSelector",
    "UniformEntryBirth",
    "UniformBirth",
    "WindowedQuota",
    "cSET",
    "cRigL",
    "cRES",
]
