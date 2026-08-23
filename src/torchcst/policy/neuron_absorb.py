"""NeuronAbsorb: row-measure merge for the neuron interface (stage 1, geometry-only).

Design is frozen in two documents, not this module's docstrings alone:
``cst/docs/research/twin-control.md`` Sec.4 ("measure としてのニューロン, 積計
量, 1受け手射影") and ``cst/tex/fast_construction_mechanics.tex`` Sec.6
("NeuronAbsorb（未実装・定義のみ）"). Read those first; this file is their
executable half.

The neuron layer's (linearized) job is pure quadrature: ``C_mk = sum_j
gamma_j * kappa(s_m, mu_j) * kappa(t_k, mu_j)`` -- a weighted point measure
``rho = sum_j gamma_j * delta_mu_j`` integrated against two kernels. A row's
gate ``gamma_j`` *is* its quadrature weight, so merging two rows is pure
gate bookkeeping (no edge to rewire, unlike a dense net's neuron merge):

    c*      = gamma_j * rho_jk                  (twin-control.md Sec.4)
    gamma_k <- gamma_k + c*
    gamma_j <- 0, row j retires

``rho_jk``, the kernel-family overlap between two rows, is the geometry factor of
the product metric ``<f_j, f_k> = rho_jk * <sigma_j, sigma_k>_D`` (tex
Sec.6.2). This court implements *only* the geometry factor -- the
activation-correlation term ``<sigma_j, sigma_k>_D`` is real-batch evidence
(an activation Gram over the two rows' downstream readouts) that has no
instrument in this codebase yet (twin-control.md's implementation table
lists it as a separate, still-unbuilt line: "活性化 Gram instrument"). Taking
the geometry-only ratio ``||f_j|| / ||f_k|| = 1`` (twin-control.md Sec.4)
turns the general one-receiver projection ``c* = gamma_j * <f_j,f_k> /
||f_k||^2`` into the plain product ``c* = gamma_j * rho_jk`` used here.  When
that instrument exists, a stage-2 court can multiply in the measured
activation correlation without changing anything else in this module (the
same ladder shape as ``twin-control.md``'s birth-side gain_perp rungs).

No multi-receiver solve. ``AbsorbCourt``/``GramService.plan_chain`` on the
synapse side already established why: a receiver set containing a twin pair
makes the local Gram near-singular and reintroduces the exact cancelling-pair
blowup absorb exists to cure (fast_construction_mechanics.tex Sec.5.3,
"一括 alpha* の自己言及的な罠"). The row-measure primitive here is one
dying row into exactly one receiver, bounded unconditionally by
Cauchy-Schwarz (``rho_jk <= 1`` under the geometry-only ratio, so ``|c*| <=
|gamma_j|`` always -- never a threshold, an identity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from torchcst._validation import require_int, require_real
from torchcst.representation.kernels import (
    OverlapScale,
    pairwise_overlap,
    require_overlap_scale,
)
from torchcst.storage import NeuronGateCredit, NeuronRetire, NeuronView

from .bundle import ProposalBundle


def _pairwise_rho(mu: Tensor, scale: OverlapScale) -> Tensor:
    """``K x K`` geometry overlap ``rho_jk`` from the kernel family.

    Same overlap as the synapse-side novelty discount
    (``policy/neuron_growth.py::_novelty_discount``) and the tex note's
    Eq. for ``rho_jk``, but the full pairwise matrix rather than
    candidate-vs-reference: the live neuron population is the fixed chart
    width, small enough that a dense ``K x K`` distance matrix is cheap
    (unlike the synapse side's screened/chunked ``GramService``, which
    exists because ``K`` there can be in the tens of thousands).

    ``scale`` resolves through :func:`~torchcst.representation.kernels.
    pairwise_overlap`, so a site whose family is not Gaussian hands its
    kernel and is priced correctly (or refused outright) instead of being
    quoted Gaussian numbers.
    """
    coordinates = mu if mu.ndim == 2 else mu.unsqueeze(-1)
    distance2 = torch.cdist(coordinates, coordinates).square()
    return pairwise_overlap(scale, distance2)


@dataclass
class NeuronAbsorbCourt:
    """Stage-1 (geometry-only) twin merge for one neuron store's live rows.

    Placed as a :class:`~torchcst.policy.families.NeuronLifecycle`
    ``absorb_factory`` -- see that module's docstring and
    :func:`~torchcst.policy.families.neuron_absorb` for why this is a
    distinct interface seat from ``retention_factory`` rather than an
    extension of :class:`~torchcst.policy.contract.RetentionCourt`: a
    retention court's contract is "return only ``NeuronRetire``"
    (enforced by :meth:`~torchcst.policy.runtime.InterfaceChild.
    decide_retention`), and NeuronAbsorb's receiver credit has no such op to
    return -- it needs its own capability, run in its own stage, exactly as
    the synapse side's absorb runs in its own stage before (not as part of)
    synapse retention.

    ``bandwidth`` sets the geometry kernel scale -- a bare float for the
    Gaussian form, or the site's own :class:`~torchcst.representation.
    ContinuousKernel` for the family-generic one (``rho_jk``'s
    ``sigma``); ``threshold`` is the minimum ``rho_jk`` for a pair to be
    considered a twin candidate at all (below it, two rows are simply
    different quadrature points and merging them would misrepresent the
    measure, not just risk it); ``rent`` is the acceptance line for
    ``cost <= rent``, matching ``AbsorbCourt``'s single-number test.

    Within one call, at most ``budget`` disjoint pairs are merged: a row
    participates in at most one merge per event, as either the dying party
    or the receiver, never both. This is a deliberate scope decision (see
    :func:`propose`'s docstring), not a limitation of the tex note's
    primitive itself -- chains of merges (a row absorbed this event, then
    itself absorbing further, later) are simply deferred to the next
    event's snapshot, keeping the store-level batch validation (see
    ``NeuronGateCredit``) simple: no id is ever both a credit target and a
    retiring row in the same commit.
    """

    bandwidth: OverlapScale
    rent: float
    threshold: float = 0.5
    requires: tuple[Any, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.bandwidth = require_overlap_scale(self.bandwidth, "bandwidth")
        self.rent = require_real(self.rent, "rent", nonnegative=True)
        self.threshold = require_real(
            self.threshold, "threshold", nonnegative=True
        )
        if self.threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        """No instruments: stage-1 similarity is a closed-form coordinate
        overlap, and acceptance reads only structural state (gate, mu) --
        the loss-blindness CLAUDE.md requires of an ordinary court."""
        del site, instruments

    def propose(
        self, view: NeuronView, budget: int, rng: torch.Generator
    ) -> tuple[ProposalBundle, ...]:
        """Propose up to ``budget`` disjoint twin-row merges.

        Candidates are every live pair with ``rho_jk >= threshold`` and
        ``cost <= rent`` (``cost = 0.5 * gamma_dying**2 * (1 - rho_jk**2)``,
        the geometry-only residual-projection cost: with ``||f_j||/||f_k||
        == 1`` at this stage, the fraction of ``f_j`` *not* explained by
        ``f_k`` is ``1 - rho_jk**2`` of its squared norm, exactly
        ``AbsorbCourt``'s ``0.5 * w**2 * res2_D`` shape carried over row-for-
        atom). Candidates are then greedily accepted cheapest-first (mirrors
        ``GramService.plan_chain``'s "pick cheapest, remove, repeat"),
        skipping any pair that shares a row with one already accepted this
        call -- the single-receiver, no-batch-solve discipline
        fast_construction_mechanics.tex Sec.5.3 requires (a receiver set
        with its own twins inside it is exactly the near-singular-Gram
        blowup absorb exists to prevent; disjoint pairs sidestep the
        question entirely rather than re-deriving that guard here).

        ``rng`` is unused (the selection is a deterministic function of
        structural state) but kept for the ``OpProposer``-shaped calling
        convention every other court/proposer in this package follows.
        """
        del rng
        require_int(budget, "budget", minimum=0)
        k_live = int(view.ids.numel())
        if budget == 0 or k_live < 2:
            return ()

        # Detached: this is a policy read of structural state, never an
        # autograd-tracked quantity (courts.py's own convention for every
        # view.mass/view.gate read).
        mu = view.mu.detach()
        if not mu.is_floating_point():
            mu = mu.to(torch.get_default_dtype())
        rho = _pairwise_rho(mu, self.bandwidth)
        rows, cols = torch.triu_indices(k_live, k_live, offset=1)
        pair_rho = rho[rows, cols]
        eligible = pair_rho >= self.threshold
        if not bool(eligible.any()):
            return ()
        rows = rows[eligible]
        cols = cols[eligible]
        pair_rho = pair_rho[eligible]

        gate = view.gate.detach()
        gate_rows = gate.index_select(0, rows)
        gate_cols = gate.index_select(0, cols)
        # The row with the smaller |gamma| dies into the larger; a tie picks
        # the lower live-view position, deterministically.
        rows_die = gate_rows.abs() <= gate_cols.abs()
        dying_pos = torch.where(rows_die, rows, cols)
        receiver_pos = torch.where(rows_die, cols, rows)
        dying_gate = torch.where(rows_die, gate_rows, gate_cols)

        cost = 0.5 * dying_gate.square() * (1.0 - pair_rho.square())
        accept = cost <= self.rent
        if not bool(accept.any()):
            return ()
        dying_pos = dying_pos[accept]
        receiver_pos = receiver_pos[accept]
        dying_gate = dying_gate[accept]
        pair_rho = pair_rho[accept]
        cost = cost[accept]

        order = torch.argsort(cost, stable=True)  # cheapest (best twin) first

        used: set[int] = set()
        chosen_dying: list[int] = []
        chosen_receiver: list[int] = []
        chosen_delta: list[float] = []
        for index in order.tolist():
            if len(chosen_dying) >= budget:
                break
            dying = int(dying_pos[index])
            receiver = int(receiver_pos[index])
            if dying in used or receiver in used:
                continue
            used.add(dying)
            used.add(receiver)
            chosen_dying.append(dying)
            chosen_receiver.append(receiver)
            chosen_delta.append(float(dying_gate[index]) * float(pair_rho[index]))

        if not chosen_dying:
            return ()

        dying_ids = view.ids.index_select(
            0, torch.tensor(chosen_dying, dtype=torch.int64)
        )
        receiver_ids = view.ids.index_select(
            0, torch.tensor(chosen_receiver, dtype=torch.int64)
        )
        delta = torch.tensor(chosen_delta, dtype=gate.dtype, device=gate.device)
        credit = NeuronGateCredit(view.site, receiver_ids, delta)
        retire = NeuronRetire(view.site, dying_ids)
        bundle = ProposalBundle(
            f"neuron_absorb:{view.site}:{view.version}", (credit, retire)
        )
        return (bundle,)
