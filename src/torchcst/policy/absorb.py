"""Stage 3b of ``docs/absorb-and-gram-design.md``: absorb policy wiring.

:class:`AbsorbCourt` is a tree-native lifecycle part: an
``OpProposer``-shaped rule (``propose(view, budget, registry, rng)``) that
:func:`~torchcst.policy.families.RENT` places as a
:class:`~torchcst.policy.families.SynapseLifecycle`'s ``absorb_factory``.
The tree is the sole authoring path for absorb (the earlier whole-policy
``AbsorbPolicy`` route is retired; see git history for its accounting).

:class:`AbsorbCourt` builds a
:class:`~torchcst.representation.gram.GramService` from the live
:class:`~torchcst.storage.synapse.SynapseView` and the
:class:`~torchcst.instruments.base.KernelPort` bound at construction (via the
public :class:`~torchcst.instruments.base.KernelPortRequest`), plans a
sequential absorb chain (:meth:`GramService.plan_chain`) capped at ``rent``,
maps the resulting live-view positions to entity IDs, and emits the ordered
:class:`~torchcst.storage.synapse.SynapseAbsorb` ops as one atomic
:class:`~torchcst.policy.bundle.ProposalBundle` -- "in list order inside one
atomic ticket" per the design's section 1, which a flat tuple of standalone
ops would *not* give (each standalone op becomes its own commit).

Acceptance is the single number ``cost <= rent``; ``alpha`` is only ever the
delivery manifest, never an acceptance input (design section 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from torchcst._validation import require_int, require_real

# Submodule import: torchcst.instruments' package __init__ may itself be
# mid-import (it reaches back into torchcst.policy), so policy modules must
# import instrument symbols from their defining modules.
from torchcst.instruments.base import KernelPort, KernelPortRequest
from torchcst.representation.gram import AbsorbPlanStep, GramService
from torchcst.storage import SynapseAbsorb, SynapseDeath, SynapseView

from .bundle import Op, ProposalBundle
from .registry import RetiredCandidateRegistry


@dataclass(frozen=True)
class AbsorbAuditEntry:
    """One committed absorb step's diagnostics (design section 5's audit list).

    ``delta_w_frobenius`` is ``||Delta W||_F == |c_dying| * sqrt(res2_F)`` --
    the double-audit Frobenius residual alongside the D-weighted training-loss
    ``cost`` (design section 1, guard 1). ``alpha_norm`` is the delivery
    manifest's magnitude, reported for diagnostics only -- never an acceptance
    input.
    """

    event_index: int
    dying: int
    receiver_count: int
    cost: float
    res2_D: float
    delta_w_frobenius: float
    alpha_norm: float


def _reassess_step(gram: GramService, step: AbsorbPlanStep) -> tuple[float, float, Tensor]:
    """Recompute ``(res2_D, res2_F, alpha)`` for one already-planned step.

    :class:`GramService.plan_chain` only returns ``(dying, receivers,
    delta_w, cost)`` per step -- not the ``res2_D``/``res2_F``/``alpha`` this
    policy's audit contract additionally requires. Gram entries depend only
    on atom positions, never amplitudes (the same fact that lets
    ``plan_chain`` simulate a whole chain off one static Gram); the exact
    ``(dying, receiver-set)`` assessment ``plan_chain`` used internally is
    therefore exactly reproducible after the fact from
    :meth:`GramService.gram_block` alone -- GramService's only public block
    accessor, so nothing here reaches into its private ``_assess``.
    """
    device = gram.device
    idx_k = torch.tensor([step.dying], dtype=torch.int64, device=device)
    neigh = step.receivers.to(device=device)
    gamma_d_kk, gamma_f_kk = gram.gram_block(idx_k, idx_k)
    gamma_d_nn, gamma_f_nn = gram.gram_block(neigh, neigh)
    gamma_d_nk, gamma_f_nk = gram.gram_block(neigh, idx_k)
    b_d = gamma_d_nk.reshape(-1)
    b_f = gamma_f_nk.reshape(-1)
    eye = torch.eye(neigh.numel(), dtype=gamma_d_nn.dtype, device=device)
    alpha = torch.linalg.solve(gamma_d_nn + gram.ridge * eye, b_d)
    res2_d = float(
        gamma_d_kk.reshape(()) - 2.0 * (alpha @ b_d) + alpha @ (gamma_d_nn @ alpha)
    )
    res2_f = float(
        gamma_f_kk.reshape(()) - 2.0 * (alpha @ b_f) + alpha @ (gamma_f_nn @ alpha)
    )
    return max(res2_d, 0.0), max(res2_f, 0.0), alpha


def _dying_amplitude(step: AbsorbPlanStep, res2_d: float, alpha: Tensor) -> float:
    """Recover ``|c_dying|`` at the moment this step was planned.

    Two independent, non-degenerate-in-different-regimes routes, matching
    the design's own case table: ``cost == 0.5 * c^2 * res2_D`` is
    well-conditioned exactly when ``res2_D`` is *not* near zero (the
    isolated/paid-prune case); ``delta_w == c * alpha`` is well-conditioned
    exactly when ``alpha`` is *not* near zero (the near-duplicate/group
    cases, where ``res2_D`` is the tiny quantity). Falls back to ``0.0`` only
    when both the residual and the delivered mass are degenerate together
    (the free-prune "no influence" case, where the reported Frobenius
    displacement is negligible regardless).
    """
    if res2_d > 1e-20:
        return (2.0 * step.cost / res2_d) ** 0.5
    alpha_norm = float(alpha.norm())
    if alpha_norm > 1e-20:
        return float(step.delta_w.norm()) / alpha_norm
    return 0.0


@dataclass
class AbsorbCourt:
    """Stage 3b: absorb as a tree-native ``SynapseLifecycle.absorb_factory`` part.

    Design section 5, stage 3b item 2: at each root-issued call it builds
    a fresh :class:`~torchcst.representation.gram.GramService` from the live
    view and the site's :class:`~torchcst.instruments.base.KernelPort`, runs
    :meth:`~torchcst.representation.gram.GramService.plan_chain` capped at
    ``rent``, and emits the ordered absorb ops as one atomic
    :class:`~torchcst.policy.bundle.ProposalBundle`.

    An atom with **no live neighbor at all** can never appear as a
    ``plan_chain`` candidate (a chain step always needs a receiver) -- but the
    design still wants it priced: distance is symmetric, so a neighborless
    atom can also never be anyone else's receiver, and is therefore untouched
    by the chain simulation. When ``include_isolated`` (default), every such
    atom whose full cost ``0.5 * c**2 * ||psi||_D**2 < rent`` is emitted as a
    plain :class:`~torchcst.storage.synapse.SynapseDeath` in the *same*
    bundle -- "receiver-less absorb IS pure prune, so a RENT policy needs no
    separate prune court" (design section 5, stage 3b item 2).

    ``budget`` caps how many chain steps :meth:`~torchcst.representation.gram.GramService.
    plan_chain` may plan in one call; the engine's own per-event
    ``StructuralQuota.synapse_absorb`` allocation (the ``budget`` argument
    :meth:`propose` receives) is a second, independent cap on the *total*
    ops this call may return (chain absorbs plus isolated deaths together) --
    whichever of the two is smaller wins for the chain, and the isolated pass
    then spends whatever total budget remains. Per the design's economy
    rule, both are plain floats/ints passed in, never a shared mutable
    object between rules.

    ``audit_delta_w`` gates only the *extra* diagnostic solve
    (:func:`_reassess_step`) needed for a chain step's ``||Delta W||_F``/
    ``||alpha||`` double-audit fields (design section 1, guard 1); it never
    changes which ops are proposed. Isolated-atom entries cost nothing extra
    to report (:meth:`~torchcst.representation.gram.GramService.residual` is
    already required to gate them) and are always recorded.
    """

    rent: float
    radius: float
    ridge: float
    budget: int | None = None
    include_isolated: bool = True
    audit_delta_w: bool = True
    data: Callable[[], Tensor] | None = None
    requires: tuple[KernelPortRequest, ...] = field(init=False, repr=False)
    audit_log: list[AbsorbAuditEntry] = field(default_factory=list, init=False)
    _request: KernelPortRequest = field(init=False, repr=False)
    _ports: dict[str, KernelPort] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rent = require_real(self.rent, "rent", nonnegative=True)
        self.radius = require_real(self.radius, "radius", positive=True)
        self.ridge = require_real(self.ridge, "ridge", nonnegative=True)
        if self.budget is not None:
            if isinstance(self.budget, bool) or not isinstance(self.budget, int):
                raise TypeError("budget must be an int or None")
            if self.budget < 0:
                raise ValueError("budget must be non-negative")
        if not isinstance(self.include_isolated, bool):
            raise TypeError("include_isolated must be a bool")
        if not isinstance(self.audit_delta_w, bool):
            raise TypeError("audit_delta_w must be a bool")
        if self.data is not None and not callable(self.data):
            raise TypeError("data must be a callable batch provider or None")
        self._request = KernelPortRequest()
        self.requires = (self._request,)

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        """Receive this site's KernelPort, built once at engine construction."""
        self._ports[site] = instruments[self._request.name].port

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[Op | ProposalBundle, ...]:
        del registry, rng
        require_int(budget, "budget", minimum=0)
        if budget == 0 or view.ids.numel() == 0:
            return ()
        gram = self._build_gram(view)
        # ``event_marker`` (the store version) stands in for the event index
        # on AbsorbAuditEntry: the court never sees the clock, and the
        # version increments exactly once per applied event.
        event_marker = int(view.version)
        chain_budget = budget if self.budget is None else min(self.budget, budget)
        ops, entries, absorbed = self._chain_ops(
            gram, view, chain_budget, event_marker
        )
        if self.include_isolated:
            isolated_ops, isolated_entries = self._isolated_deaths(
                gram, view, budget - len(ops), absorbed, event_marker
            )
            ops.extend(isolated_ops)
            entries.extend(isolated_entries)
        if not ops:
            return ()
        self.audit_log.extend(entries)
        bundle = ProposalBundle(f"absorb:{view.site}:{event_marker}", tuple(ops))
        return (bundle,)

    def _build_gram(self, view: SynapseView) -> GramService:
        """One fresh position-only Gram over the live view, per court call."""
        port = self._ports.get(view.site)
        if port is None:
            raise RuntimeError(
                f"AbsorbCourt has no KernelPort bound for site {view.site!r}; "
                "was the engine constructed with a matching compute module?"
            )
        k_in, k_out = port.columns(view.s, view.t)
        batch = None
        if self.data is not None:
            batch = self.data()
            if not isinstance(batch, Tensor):
                raise TypeError("data provider must return a Tensor")
        return GramService(
            k_out,
            k_in,
            view.w,
            view.s,
            view.t,
            radius=self.radius,
            ridge=self.ridge,
            data=batch,
        )

    def _chain_ops(
        self,
        gram: GramService,
        view: SynapseView,
        chain_budget: int,
        event_marker: int,
    ) -> tuple[list[Op], list[AbsorbAuditEntry], set[int]]:
        """Plan the rent-capped absorb chain and map positions to entity IDs."""
        steps = gram.plan_chain(budget=chain_budget, cost_cap=self.rent)
        ops: list[Op] = []
        entries: list[AbsorbAuditEntry] = []
        absorbed_positions: set[int] = set()
        for step in steps:
            absorbed_positions.add(step.dying)
            # ``step.receivers`` are live-view positions on the *atoms'*
            # device, while ``view.ids`` is the store's host-side id tensor:
            # the lookup must happen on the id tensor's device, or
            # ``index_select`` raises on any accelerator.
            receivers = view.ids.index_select(
                0, step.receivers.to(view.ids.device)
            )
            ops.append(
                SynapseAbsorb(
                    site=view.site,
                    dying=int(view.ids[step.dying]),
                    receivers=receivers,
                    delta_w=step.delta_w,
                )
            )
            if self.audit_delta_w:
                res2_d, res2_f, alpha = _reassess_step(gram, step)
                c_dying = _dying_amplitude(step, res2_d, alpha)
                entries.append(
                    AbsorbAuditEntry(
                        event_index=event_marker,
                        dying=int(view.ids[step.dying]),
                        receiver_count=int(step.receivers.numel()),
                        cost=step.cost,
                        res2_D=res2_d,
                        delta_w_frobenius=c_dying * (res2_f ** 0.5),
                        alpha_norm=float(alpha.norm()),
                    )
                )
        return ops, entries, absorbed_positions

    def _isolated_deaths(
        self,
        gram: GramService,
        view: SynapseView,
        remaining: int,
        absorbed_positions: set[int],
        event_marker: int,
    ) -> tuple[list[Op], list[AbsorbAuditEntry]]:
        """Price neighborless atoms as receiver-less absorbs (pure prunes)."""
        ops: list[Op] = []
        entries: list[AbsorbAuditEntry] = []
        for position in range(gram.k_live):
            if remaining <= 0:
                break
            if position in absorbed_positions:
                continue
            if gram.neighbors(position).numel():
                continue
            assessment = gram.residual(position)
            full_cost = 0.5 * float(gram.w[position]) ** 2 * assessment.res2_D
            if full_cost >= self.rent:
                continue
            ops.append(SynapseDeath(view.site, view.ids[position].reshape(1)))
            entries.append(
                AbsorbAuditEntry(
                    event_index=event_marker,
                    dying=int(view.ids[position]),
                    receiver_count=0,
                    cost=full_cost,
                    res2_D=assessment.res2_D,
                    delta_w_frobenius=0.0,
                    alpha_norm=0.0,
                )
            )
            remaining -= 1
        return ops, entries
