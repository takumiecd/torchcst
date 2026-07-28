"""Stage 3a of ``docs/absorb-and-gram-design.md``: whole-policy absorb wiring.

:class:`AbsorbPolicy` is a first-class :class:`~torchcst.policy.contract.
StructuralPolicy` (``docs/policy-authoring.md``, authoring path 2): one object
implementing ``capture``, ``plan``, ``bind_instruments``, and ``on_applied``,
with no cadence, distributor, or court decomposition. At each structural
event it builds a :class:`~torchcst.representation.gram.GramService` from the
live :class:`~torchcst.storage.synapse.SynapseView` and the
:class:`~torchcst.instruments.base.KernelPort` bound at construction, plans a
sequential absorb chain (:meth:`GramService.plan_chain`) capped at ``rent``,
maps the resulting live-view positions to entity IDs, and emits the ordered
:class:`~torchcst.storage.synapse.SynapseAbsorb` ops as one atomic
:class:`~torchcst.policy.bundle.ProposalBundle` -- "in list order inside one
atomic ticket" per the design's section 1, which a flat tuple of standalone
ops passed through :meth:`~torchcst.engine.StructuralEngine.apply_proposals`
would *not* give (each standalone op there becomes its own commit).

Acceptance is the single number ``cost <= rent``; ``alpha`` is only ever the
delivery manifest, never an acceptance input (design section 1). This policy
never reads an objective anywhere -- ``capture`` always returns ``False``, so
no backward statistic is ever measured -- keeping it loss-blind as the design
requires. The only reason this module declares an ``ObservationRequest`` at
all is that the framework's ``requires``/``bind_instruments`` channel is the
sole sanctioned way for a whole policy to reach a compute module's
``kernel_columns()`` (see ``docs/policy-authoring.md``'s "declare and bind
observations" pattern); the instrument built from that request never
measures anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Callable

import torch
from torch import Tensor

from torchcst.instruments import InstrumentBuildContext, KernelPort
from torchcst.representation.gram import AbsorbPlanStep, GramService
from torchcst.storage import SynapseAbsorb

from .bundle import ProposalBundle
from .contract import Clock, PolicyContext, StructuralPlan


@dataclass(frozen=True)
class _KernelPortRequest:
    """Trivial observation request whose only purpose is delivering a KernelPort.

    ``AbsorbPolicy.capture`` always returns ``False``, so the engine never
    enables capture and this request's instrument is never asked to measure
    anything; ``measure_after_backward`` below exists only to satisfy the
    ``DeferredCaptureInstrument`` structural check
    :class:`~torchcst.engine.StructuralEngine` performs unconditionally for
    every declared requirement at construction time.
    """

    name: str
    timing: str = "after_backward"

    def build(self, context: InstrumentBuildContext) -> "_KernelPortInstrument":
        return _KernelPortInstrument(context.module)


class _KernelPortInstrument:
    """Capture-instrument shell whose only state is a read-only KernelPort."""

    def __init__(self, module: Any) -> None:
        if not callable(getattr(module, "kernel_columns", None)):
            raise TypeError("AbsorbPolicy requires a compute module with kernel_columns()")
        self.name = "kernel_port"
        self.port = KernelPort(module)

    def prepare(self, view: Any, module: Any) -> None:
        del view, module

    def measure_after_backward(self, module: Any, x: Tensor, g_out: Tensor) -> Any:
        del module, x, g_out
        raise RuntimeError(
            "AbsorbPolicy never captures backward observations; this "
            "instrument should never be asked to measure anything"
        )

    def finalize_update(self, measurements: Any, view: Any) -> None:
        del measurements, view


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


def _validate_real(value: Any, name: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if allow_zero and value < 0:
        raise ValueError(f"{name} must be non-negative")
    if not allow_zero and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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
class AbsorbPolicy:
    """Whole StructuralPolicy: GramService absorb chains at a fixed cadence.

    ``site`` names the one :class:`~torchcst.storage.synapse.SynapseStore`
    this policy acts on. ``event_interval`` is the cadence (an absorb event
    is attempted every ``event_interval`` updates, mirroring
    :class:`~torchcst.policy.cadences.PeriodicCadence`'s field name).
    ``rent`` is the fixed cost cap (``GramService.plan_chain``'s
    ``cost_cap``) -- a threshold parameter, never a measured profit: this
    policy has no objective access anywhere. ``radius``/``ridge`` configure
    the ``GramService`` built fresh from the live view at each event.
    ``budget`` optionally caps the number of absorbs planned per event
    (``None`` means unlimited, i.e. plan until no eligible candidate
    remains). ``data`` is an optional zero-argument batch provider called
    once per event to supply ``GramService``'s ``[n, n_in]`` D-metric batch;
    the identity metric is used when it is ``None``.
    """

    site: str
    event_interval: int
    rent: float
    radius: float
    ridge: float
    budget: int | None = None
    data: Callable[[], Tensor] | None = None
    requires: tuple[_KernelPortRequest, ...] = field(init=False, repr=False)
    audit_log: list[AbsorbAuditEntry] = field(default_factory=list, init=False)
    _request: _KernelPortRequest = field(init=False, repr=False)
    _port: KernelPort | None = field(default=None, init=False, repr=False)
    _pending_ops: tuple[SynapseAbsorb, ...] = field(
        default=(), init=False, repr=False
    )
    _pending_entries: tuple[AbsorbAuditEntry, ...] = field(
        default=(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or not self.site:
            raise ValueError("site must be a non-empty string")
        if isinstance(self.event_interval, bool) or not isinstance(
            self.event_interval, int
        ):
            raise TypeError("event_interval must be an int")
        if self.event_interval <= 0:
            raise ValueError("event_interval must be positive")
        self.rent = _validate_real(self.rent, "rent")
        self.radius = _validate_real(self.radius, "radius", allow_zero=False)
        self.ridge = _validate_real(self.ridge, "ridge")
        if self.budget is not None:
            if isinstance(self.budget, bool) or not isinstance(self.budget, int):
                raise TypeError("budget must be an int or None")
            if self.budget < 0:
                raise ValueError("budget must be non-negative")
        if self.data is not None and not callable(self.data):
            raise TypeError("data must be a callable batch provider or None")
        self._request = _KernelPortRequest(name=f"absorb_kernel_port::{self.site}")
        self.requires = (self._request,)

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        """Receive this site's KernelPort, built once at engine construction."""
        if site != self.site:
            return
        self._port = instruments[self._request.name].port

    def capture(self, clock: Clock) -> bool:
        """This policy never observes backward statistics; it is loss-blind."""
        del clock
        return False

    def plan(self, context: PolicyContext) -> StructuralPlan | None:
        if context.clock.update_step == 0 or context.clock.update_step % self.event_interval:
            return None
        if self._port is None:
            raise RuntimeError(
                f"AbsorbPolicy has no KernelPort bound for site {self.site!r}; "
                "was the engine constructed with a matching compute module?"
            )
        view = context.synapses[self.site]
        if view.ids.numel() < 2:
            # Quiescent: fewer than two live atoms means no possible receiver.
            self._pending_ops = ()
            self._pending_entries = ()
            return StructuralPlan()

        k_in, k_out = self._port.columns(view.s, view.t)
        batch = None
        if self.data is not None:
            batch = self.data()
            if not isinstance(batch, Tensor):
                raise TypeError("data provider must return a Tensor")
        gram = GramService(
            k_out,
            k_in,
            view.w,
            view.s,
            view.t,
            radius=self.radius,
            ridge=self.ridge,
            data=batch,
        )
        steps = gram.plan_chain(budget=self.budget, cost_cap=self.rent)
        if not steps:
            # Quiescent: nothing is below rent at this event.
            self._pending_ops = ()
            self._pending_entries = ()
            return StructuralPlan()

        ops: list[SynapseAbsorb] = []
        entries: list[AbsorbAuditEntry] = []
        event_index = context.clock.event_index
        for step in steps:
            res2_d, res2_f, alpha = _reassess_step(gram, step)
            c_dying = _dying_amplitude(step, res2_d, alpha)
            ops.append(
                SynapseAbsorb(
                    site=self.site,
                    dying=int(view.ids[step.dying]),
                    receivers=view.ids.index_select(0, step.receivers),
                    delta_w=step.delta_w,
                )
            )
            entries.append(
                AbsorbAuditEntry(
                    event_index=event_index,
                    dying=int(view.ids[step.dying]),
                    receiver_count=int(step.receivers.numel()),
                    cost=step.cost,
                    res2_D=res2_d,
                    delta_w_frobenius=c_dying * (res2_f ** 0.5),
                    alpha_norm=float(alpha.norm()),
                )
            )
        self._pending_ops = tuple(ops)
        self._pending_entries = tuple(entries)
        bundle = ProposalBundle(f"absorb:{event_index}", tuple(ops))
        return StructuralPlan((bundle,))

    def on_applied(
        self, context: PolicyContext, operations: tuple[Any, ...]
    ) -> None:
        """Record audit diagnostics only for absorbs that actually committed.

        The whole planned chain is one atomic :class:`ProposalBundle`: either
        every planned op committed (``operations`` then holds exactly the
        same op *objects* this policy emitted, in order -- the engine's
        prepare/commit path never reconstructs
        ``SynapseAbsorb``/``SynapseBirth``/``SynapseDeath`` instances) or the
        bundle's ``prepare()`` failed atomically and none did. Ops are
        compared by identity, not ``==``: a dataclass ``__eq__`` over
        ``SynapseAbsorb``'s Tensor fields would raise on any multi-element
        ``receivers``/``delta_w``, since tuple comparison forces each paired
        field through ``bool()``.
        """
        del context
        pending_ops, pending_entries = self._pending_ops, self._pending_entries
        self._pending_ops = ()
        self._pending_entries = ()
        applied = tuple(
            op
            for op in operations
            if isinstance(op, SynapseAbsorb) and op.site == self.site
        )
        if len(applied) != len(pending_ops) or any(
            a is not b for a, b in zip(applied, pending_ops)
        ):
            return
        self.audit_log.extend(pending_entries)
