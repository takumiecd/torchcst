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
from .cadences import BirthWindowCadence, PeriodicCadence
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
    QuotaPolicy,
    RetentionCourt,
    StructuralQuota,
)
from .courts import MagnitudeCourt, RentCourt
from .families import (
    RENT,
    NeuronLifecycle,
    SynapseLifecycle,
    cRES,
    cRigL,
    cSET,
    cSFW,
    cVP,
    gamma_ungate,
    thinned,
)
from .profit import ProfitCourt, TrialSession, TrialTransaction
from .proposers import (
    GradFieldTopKBirth,
    IncidentOutputBirth,
    MergeProposer,
    OrthogonalBirth,
    UniformBirth,
    UniformEntryBirth,
)
from .quotas import CallableQuota, ConstantQuota, QuotaWindow, WindowedQuota
from .registry import RetiredCandidateRegistry
from .runtime import (
    EndpointChild,
    InterfaceChild,
    PricedProposal,
    SiteBinding,
    SynapseChild,
)
from .scored import ScoredBirth, TopKSelector
from .tree import QuotaRegime, RentEconomy, RuntimeTree
from . import recipes

__all__ = [
    "AbsorbAuditEntry",
    "AbsorbCourt",
    "BirthWindowCadence",
    "BudgetDistributor",
    "BudgetRequest",
    "BundleComposer",
    "Cadence",
    "CallableQuota",
    "Clock",
    "ConstantQuota",
    "EndpointChild",
    "EvenBudgetDistributor",
    "EventSignal",
    "GradFieldTopKBirth",
    "IncidentOutputBirth",
    "InstrumentSpec",
    "InterfaceChild",
    "MagnitudeCourt",
    "MergeProposer",
    "NeuronLifecycle",
    "ObservationRequest",
    "Op",
    "OpProposer",
    "OrthogonalBirth",
    "PeriodicCadence",
    "Phase",
    "PricedProposal",
    "ProfitCourt",
    "ProposalBundle",
    "QuotaPolicy",
    "QuotaRegime",
    "QuotaWindow",
    "RENT",
    "RentCourt",
    "RentEconomy",
    "RetentionCourt",
    "RetiredCandidateRegistry",
    "RuntimeTree",
    "ScoredBirth",
    "SiteBinding",
    "StructuralQuota",
    "SynapseChild",
    "SynapseLifecycle",
    "TopKSelector",
    "TrialSession",
    "TrialTransaction",
    "UniformBirth",
    "UniformEntryBirth",
    "WindowedQuota",
    "bundle_birth_count",
    "cRES",
    "cRigL",
    "cSET",
    "cSFW",
    "cVP",
    "gamma_ungate",
    "recipes",
    "thinned",
]
