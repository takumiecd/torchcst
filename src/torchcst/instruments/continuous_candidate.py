"""Continuous-chart candidate scoring instrument (cRigL/cRES).

Reading order: linear-algebra helpers, :class:`RefinementSchedule`,
:class:`ContinuousCandidateField`, :class:`ContinuousCandidateRequest`.
Pool sampling (uniform / hardcore / local refinement draws) lives in
:mod:`torchcst.instruments._candidate_sampling`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import Tensor

from torchcst._validation import require_int
from torchcst.compute import ObservationTiming, flatten_capture_pair
from torchcst.representation import Box
from torchcst.storage import SynapseStore, SynapseView

# _coarse_spacing is re-exported here for external callers of the private API.
from ._candidate_sampling import PoolSampler, _coarse_spacing
from .base import (
    CandidateSnapshot,
    FactorPort,
    InstrumentBuildContext,
    WeightedMeasurement,
    weighted_sum,
)

__all__ = [
    "ContinuousCandidateField",
    "ContinuousCandidateRequest",
    "RefinementSchedule",
]


def _normalize_rows(x: Tensor, eps: float) -> Tensor:
    """Row-normalize ``x``, mapping any near-zero row to the zero vector.

    A near-zero row means the candidate/live atom's factor column vanishes
    (e.g. a compact-support factor evaluated far from every neuron); such a
    row has no well-defined direction, so it is left at zero rather than
    blown up by dividing by a tiny norm.
    """
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    safe = norm.clamp_min(eps)
    unit = x / safe
    return torch.where(norm > eps, unit, torch.zeros_like(unit))


def _solve_gram(gamma: Tensor, rhs: Tensor, eps: float) -> Tensor:
    """Solve ``gamma @ x = rhs``, falling back to a least-squares solve.

    Near-duplicate live atoms make ``gamma`` (the live-atom direction Gram
    matrix) singular or numerically unstable in practice.  A small diagonal
    jitter is tried first through ``torch.linalg.solve``; if that raises or
    returns a non-finite result, the minimum-norm least-squares solution via
    ``torch.linalg.pinv`` (SVD-based) is the guaranteed-finite fallback for a
    singular/rank-deficient ``gamma``.  ``torch.linalg.lstsq`` would be the
    more direct fallback, but it has no MPS factor (hard error, not just a
    slow fallback) as of this torch version; ``pinv`` gives the identical
    solution for a symmetric PSD ``gamma`` and MPS transparently falls back
    to CPU for it.
    """
    k = gamma.shape[0]
    jitter = eps * torch.eye(k, dtype=gamma.dtype, device=gamma.device)
    try:
        solution = torch.linalg.solve(gamma + jitter, rhs)
        if bool(torch.isfinite(solution).all()):
            return solution
    except torch.linalg.LinAlgError:
        pass
    return torch.linalg.pinv(gamma) @ rhs


@dataclass(frozen=True)
class RefinementSchedule:
    """Coarse-to-fine schedule layered on top of ``ContinuousCandidateField``.

    Discrete RigL takes an *exact* argmax over its dormant candidate set --
    there is no sampling involved, every candidate is enumerated. The
    faithful continuous counterpart of "exact argmax" is an argmax over the
    whole chart (a continuum), which no finite pool can enumerate; a single
    uniform pool only *approximates* that argmax, and Stage 0 measured that
    approximation to be too coarse at any pool size a materialised
    ``[M, in_features]`` factor-column buffer can afford (M=4096 candidate
    spacing ~1.3x sigma on the worst 3D input chart; halving that needs
    ~18x the candidates in 3D, i.e. ~180 MB/layer). Coarse-to-fine
    refinement is the fix that keeps the approximation faithful without
    that memory blowup: sample a coarse pool (exactly today's uniform
    pool), score it, take the ``top_k`` highest-scoring candidates, sample
    ``samples_per_winner`` fresh candidates in a local box around each
    winner, rescore, and repeat for ``rounds`` rounds -- concentrating the
    off-grid search budget where the field is actually large instead of
    spending it uniformly everywhere. Off-grid refinement close to a
    provisional winner is exactly the degree of freedom a continuous chart
    has that a discrete dormant set does not, so this is recorded here as
    the CST-specific capability being exercised, not a liberty taken with
    the RigL analogy.

    Attributes:
        top_k: number of highest-scoring candidates to refine around, each
            round. Should be at least the largest birth budget a caller
            expects to draw from the resulting snapshot, since refinement
            only improves resolution near these winners.
        samples_per_winner: fresh local candidates drawn around each winner,
            each round.
        radius_multiplier: the local refinement box has half-width
            ``radius_multiplier`` times the coarse pool's own expected
            nearest-neighbour spacing (``domain width * pool_size **
            (-1/dim)``) -- "on the order of the coarse spacing" per round 1;
            later rounds shrink the radius to the *previous* round's local
            box density, so the search keeps zooming in. Coordinates are
            always clamped back into the chart's ``Box`` domain, so a
            winner near the boundary gets a truncated (not reflected) box.
        rounds: number of refine-and-rescore passes. All rounds' candidates
            (coarse and every refined batch) are kept and returned together,
            scored on a common footing, so the caller's top-k selection sees
            the best candidate found across every round.
    """

    top_k: int
    samples_per_winner: int
    radius_multiplier: float = 1.0
    rounds: int = 1

    def __post_init__(self) -> None:
        require_int(self.top_k, "top_k", minimum=1)
        require_int(self.samples_per_winner, "samples_per_winner", minimum=1)
        require_int(self.rounds, "rounds", minimum=1)
        if not isfinite(float(self.radius_multiplier)) or float(self.radius_multiplier) <= 0:
            raise ValueError("radius_multiplier must be a positive finite float")
        object.__setattr__(self, "radius_multiplier", float(self.radius_multiplier))


class ContinuousCandidateField:
    """Score a freshly sampled continuous candidate pool (cRigL/cRES).

    Samples ``pool_size`` fresh ``(source, target)`` candidates from the
    layer's input/output ``Box`` domains whenever the structural version
    changes, accumulates ``G`` (the batch gradient w.r.t. the materialised
    weight, ``[out_features, in_features]``) via the same
    accumulate-until-consumed backward-capture pattern used by
    :class:`~torchcst.instruments.CertificateSubspace`, and scores each
    candidate ``z = (s, t)`` against the direction ``a_z = u(t) v(s)^T``
    (Frobenius-unit-norm, ``u`` from ``factor_out``/target and ``v`` from
    ``factor_in``/source -- the same orientation :meth:`dense_weight` uses):

    * ``mode="raw"`` (cRigL): ``score(z) = <G, a_z>``.
    * ``mode="deflated"`` (cRES): the component of ``<G, a_z>`` orthogonal to
      the span of the live atoms' directions, normalized by the residual
      direction norm -- i.e. deflating the score by how much of ``a_z`` is
      already representable by the current live support.

    All factor evaluation goes through :class:`~torchcst.instruments.base.FactorPort`
    rather than reaching into the compute module's factor/neuron internals.

    ⚠ **The scores this instrument reports are signed**, unlike
    :class:`~torchcst.instruments.GradientField` and
    :class:`~torchcst.instruments.ScoredCandidates`, which already emit
    ``|grad|``. A synapse atom's coefficient is a signed scalar and the local
    profile gain is quadratic in the score, so callers must rank by
    ``|score|`` -- which is what :class:`~torchcst.policy.TopKSelector` does
    under its ``candidate_cone="signed"`` default, and what :meth:`_refine`
    does when picking which candidates to zoom in on.

    ``refinement``, when given a :class:`RefinementSchedule`, layers
    coarse-to-fine off-grid refinement on top of the uniform ``pool_size``
    pool (see that class's docstring for why this is the faithful reading
    of continuous RigL/RES rather than a liberty). ``pool_size`` is then the
    *coarse* pool: :meth:`candidate_snapshot` scores it, refines locally
    around the top-scoring candidates, and returns the union of coarse and
    refined candidates, all scored on a common footing. Refinement runs
    inside :meth:`candidate_snapshot` (not :meth:`prepare`) because it needs
    scores, and scores need ``G``, which is only complete after backward;
    consequently -- unlike the coarse pool sampled once per structural
    version in :meth:`prepare` -- the refined candidates are recomputed,
    consuming fresh ``rng`` draws, on every :meth:`candidate_snapshot` call.
    Production usage (:class:`~torchcst.policy.ScoredBirth`) calls
    :meth:`candidate_snapshot` exactly once per structural step, so this
    does not multiply refinement cost; it does mean a diagnostic caller that
    invokes :meth:`candidate_snapshot` more than once per step will get a
    freshly re-refined (and hence not source/target-identical) pool each
    time when refinement is enabled, whereas the coarse-only pool is stable
    across such calls.

    **Certificate convention: R_{t-1}=0.** ``G`` accumulates the signed,
    weighted microbatch sum *within* one observation window, but it is
    consumed -- not merely observed -- by the structural event that reads
    it: :class:`~torchcst.engine.StructuralEngine` resets every bound
    ``ContinuousCandidateField`` (alongside every
    :class:`~torchcst.instruments.CertificateSubspace`) at the end of each
    structural event that actually applies, regardless of whether that
    specific instrument's scores were the ones a proposer used. This is the
    theory's "the residual before this window is treated as zero"
    convention: each event's :meth:`candidate_snapshot` reads a certificate
    accumulated purely since the *previous* event, never a cumulative,
    pre-optimizer-step sum spanning multiple events (the defect
    ``docs/absorb-and-gram-design.md``'s Stage 3b-fix section describes). A
    caller that never reaches a structural event (e.g. only ever calls
    :meth:`candidate_snapshot` diagnostically) must call :meth:`reset`
    explicitly at its own event boundary.
    """

    def __init__(
        self,
        store: SynapseStore,
        module: Any,
        rng: torch.Generator,
        *,
        pool_size: int,
        mode: str,
        sampling: str,
        sigma: float | None,
        hardcore_attempts: int,
        eps: float,
        name: str,
        refinement: RefinementSchedule | None = None,
    ) -> None:
        if not isinstance(store.spec.domain_in, Box) or not isinstance(
            store.spec.domain_out, Box
        ):
            raise TypeError("ContinuousCandidateField requires Box domains")
        if not callable(getattr(module, "factor_columns", None)):
            raise TypeError("compute module must provide factor_columns()")
        if refinement is not None and not isinstance(refinement, RefinementSchedule):
            raise TypeError("refinement must be a RefinementSchedule or None")
        self.name = name
        self.store = store
        self.module = module
        self.port = FactorPort(module)
        self.rng = rng
        self.pool_size = pool_size
        self.mode = mode
        self.sampling = sampling
        self.sigma = sigma
        self.hardcore_attempts = hardcore_attempts
        self.eps = eps
        self.refinement = refinement
        self._sampler = PoolSampler(
            domain_in=store.spec.domain_in,
            domain_out=store.spec.domain_out,
            rng=rng,
            like_source=store.s,
            like_target=store.t,
            sampling=sampling,
            sigma=sigma,
            attempts=hardcore_attempts,
        )
        self._version = -1
        self._source = store.s.detach().new_zeros((0, store.d_in))
        self._target = store.t.detach().new_zeros((0, store.d_out))
        self._G: Tensor | None = None

    # ---- capture lifecycle -----------------------------------------------

    def prepare(self, view: SynapseView, module: Any) -> None:
        """Resample the candidate pool only after the structural version changes."""
        if module is not self.module:
            raise ValueError("instrument was prepared with another compute module")
        if view.version == self._version:
            return
        self._source, self._target = self._sampler.pool(
            view.s, view.t, self.pool_size
        )
        self._version = view.version

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        if module is not self.module:
            raise ValueError("instrument measured another compute module")
        x_flat, g_flat = flatten_capture_pair(x, g_out, x.shape[-1], g_out.shape[-1])
        return {"matrix": g_flat.transpose(0, 1) @ x_flat}

    def reduce_backward(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        """Reduce one signed ``g_out.T @ x`` contribution inside backward."""
        return self._measure(module, x, g_out)

    def measure_after_backward(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        """Measure one signed ``g_out.T @ x`` contribution after backward."""
        return self._measure(module, x, g_out)

    def finalize_update(
        self, measurements: tuple[WeightedMeasurement, ...], view: SynapseView
    ) -> None:
        """Accumulate the weighted signed ``G`` contribution at the update boundary."""
        del view
        contribution = weighted_sum(measurements, "matrix")
        if contribution is None:
            return
        if self._G is None:
            self._G = contribution.detach().clone()
        else:
            if self._G.shape != contribution.shape:
                raise ValueError("candidate field gradient dimensions changed")
            self._G = (self._G.to(contribution) + contribution).detach()

    def reset(self) -> None:
        """Begin the next observation window without changing shape/device.

        This is the consumption half of the R_{t-1}=0 certificate convention
        (see the class docstring): the engine calls it at the end of every
        applied structural event; a caller driving this instrument outside
        the engine must call it at its own event boundary.
        """
        if self._G is not None:
            self._G.zero_()

    def raw_gradient(self) -> Tensor:
        """Return the currently accumulated certificate ``G``, in raw units.

        Same zero-fallback shape as :meth:`candidate_snapshot` uses
        internally for scoring (``[out_features, in_features]``, matching
        ``self.module``). Unlike :meth:`_score`'s Frobenius-unit-normalized
        candidate directions, this is ``G`` in its native loss-gradient
        units -- exactly what
        :class:`~torchcst.policy.scored.ScoredBirth`'s entrance-side
        loss-unit settlement needs to compute ``<G, psi>_F`` against an
        *un-normalized* candidate atom matrix ``psi``.
        """
        base = self.store.s.detach()
        if self._G is None:
            return base.new_zeros((self.module.out_features, self.module.in_features))
        return self._G.detach().to(base)

    # ---- scoring ---------------------------------------------------------

    def _score(
        self,
        source: Tensor,
        target: Tensor,
        gradient: Tensor,
        live_source: Tensor,
        live_target: Tensor,
    ) -> Tensor:
        """Score one arbitrary ``(source, target)`` pool against ``gradient``.

        Both scoring modes go through this one path, so ``mode="raw"``
        (cRigL) and ``mode="deflated"`` (cRES) stay comparable under
        refinement too.
        """
        k_live = live_source.shape[0]
        k_in_pool, k_out_pool = self.port.columns(source, target)
        u_pool = _normalize_rows(k_out_pool.transpose(0, 1), self.eps)
        v_pool = _normalize_rows(k_in_pool.transpose(0, 1), self.eps)

        raw = (u_pool * (v_pool @ gradient.transpose(0, 1))).sum(dim=1)

        if self.mode == "raw" or k_live == 0:
            return raw

        k_in_live, k_out_live = self.port.columns(live_source, live_target)
        u_live = _normalize_rows(k_out_live.transpose(0, 1), self.eps)
        v_live = _normalize_rows(k_in_live.transpose(0, 1), self.eps)

        live_scores = (u_live * (v_live @ gradient.transpose(0, 1))).sum(dim=1)
        pool_live_overlap = (u_pool @ u_live.transpose(0, 1)) * (
            v_pool @ v_live.transpose(0, 1)
        )
        gram = (u_live @ u_live.transpose(0, 1)) * (v_live @ v_live.transpose(0, 1))

        coeffs = _solve_gram(gram, pool_live_overlap.transpose(0, 1), self.eps)  # [K, M]
        proj_self = (pool_live_overlap * coeffs.transpose(0, 1)).sum(dim=1)
        proj_grad = coeffs.transpose(0, 1) @ live_scores

        residual = (1.0 - proj_self).clamp_min(0.0)
        valid = residual > self.eps
        safe_residual = residual.clamp_min(self.eps)
        return torch.where(
            valid, (raw - proj_grad) / safe_residual.sqrt(), torch.zeros_like(raw)
        )

    def _refine(
        self,
        source: Tensor,
        target: Tensor,
        scores: Tensor,
        gradient: Tensor,
        live_source: Tensor,
        live_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Coarse-to-fine loop: top-k by ``|score|``, resample locally, rescore, repeat.

        See :class:`RefinementSchedule` for the rationale and the radius
        rule. Every round's candidates (coarse pool plus every refined
        batch) are concatenated and returned together, all scored on the
        common :meth:`_score` footing, so the caller's own top-k selection
        (not this method) picks the final "best overall" -- this keeps the
        method correct for any downstream birth budget, not just budget 1.

        Winners are ranked by ``|score|``, matching
        :class:`~torchcst.policy.TopKSelector`'s ``candidate_cone="signed"``
        default and for the same reason: a synapse atom's coefficient is a
        signed scalar, the profile gain is quadratic in the score, so the
        extrema of this field are its ``|score|`` peaks in both directions.
        Refining around ``+score`` peaks only would resolve half the field
        and hand the other half to the coarse grid -- and the refined pool is
        what the caller then selects over, so the two rankings have to agree
        or refinement zooms somewhere selection will not buy.
        """
        schedule = self.refinement
        assert schedule is not None
        domain_in = self.store.spec.domain_in
        domain_out = self.store.spec.domain_out
        radius_in = _coarse_spacing(domain_in, self.pool_size) * schedule.radius_multiplier
        radius_out = _coarse_spacing(domain_out, self.pool_size) * schedule.radius_multiplier

        def shrunk(radius: float, dim: int) -> float:
            # The next round searches inside the box just sampled, which
            # holds `samples_per_winner` points -- that box's own density,
            # not the original coarse pool's, sets how far the next round
            # may look, so the search keeps zooming in round over round.
            return (
                2.0
                * radius
                * (schedule.samples_per_winner ** (-1.0 / dim))
                * schedule.radius_multiplier
            )

        for _ in range(schedule.rounds):
            top_k = min(schedule.top_k, scores.numel())
            if top_k == 0:
                break
            winners = torch.argsort(scores.abs(), descending=True, stable=True)[:top_k]
            refined_source, refined_target = self._sampler.local(
                source.index_select(0, winners),
                target.index_select(0, winners),
                schedule.samples_per_winner,
                radius_in,
                radius_out,
                live_source,
                live_target,
            )
            refined_scores = self._score(
                refined_source, refined_target, gradient, live_source, live_target
            )
            source = torch.cat((source, refined_source), dim=0)
            target = torch.cat((target, refined_target), dim=0)
            scores = torch.cat((scores, refined_scores), dim=0)
            radius_in = shrunk(radius_in, domain_in.dim)
            radius_out = shrunk(radius_out, domain_out.dim)
        return source, target, scores

    def candidate_snapshot(self) -> CandidateSnapshot:
        """Score the current candidate pool against the accumulated ``G``.

        When ``self.refinement`` is set, the returned snapshot is the union
        of the coarse pool and every round's locally-refined candidates
        (see :class:`RefinementSchedule` and :meth:`_refine`); otherwise it
        is exactly the coarse pool, unchanged from the pre-refinement
        behaviour.
        """
        view = self.store.view()
        live_source = view.s.detach()
        live_target = view.t.detach()

        gradient = self.raw_gradient()

        source = self._source
        target = self._target
        scores = self._score(source, target, gradient, live_source, live_target)

        if self.refinement is not None:
            source, target, scores = self._refine(
                source, target, scores, gradient, live_source, live_target
            )

        return CandidateSnapshot(
            source.detach().clone(),
            target.detach().clone(),
            scores.detach(),
        )

    # ---- serialization ---------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "torchcst-continuous-candidate-field-v1",
            "version": self._version,
            "source": self._source.detach().clone(),
            "target": self._target.detach().clone(),
            "G": None if self._G is None else self._G.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != "torchcst-continuous-candidate-field-v1"
        ):
            raise ValueError("unsupported ContinuousCandidateField state schema")
        self._version = int(state["version"])
        self._source = state["source"].detach().clone()
        self._target = state["target"].detach().clone()
        gradient = state.get("G")
        self._G = None if gradient is None else gradient.detach().clone()


@dataclass(frozen=True)
class ContinuousCandidateRequest:
    """Build one :class:`ContinuousCandidateField` per requested site."""

    pool_size: int = 4096
    mode: str = "raw"
    sampling: str = "uniform"
    sigma: float | None = None
    hardcore_attempts: int = 8
    eps: float = 1e-6
    name: str = "continuous_candidate_field"
    timing: ObservationTiming | str = ObservationTiming.AFTER_BACKWARD
    refinement: RefinementSchedule | None = None

    def __post_init__(self) -> None:
        self._validate_fields()
        self._validate_cross_field()
        object.__setattr__(self, "sigma", None if self.sigma is None else float(self.sigma))
        object.__setattr__(self, "eps", float(self.eps))
        try:
            resolved_timing = ObservationTiming(self.timing)
        except ValueError as exc:
            raise ValueError("timing must be backward_inline or after_backward") from exc
        object.__setattr__(self, "timing", resolved_timing)

    def _validate_fields(self) -> None:
        require_int(self.pool_size, "pool_size", minimum=1)
        if self.mode not in {"raw", "deflated"}:
            raise ValueError("mode must be 'raw' or 'deflated'")
        if self.sampling not in {"uniform", "hardcore"}:
            raise ValueError("sampling must be 'uniform' or 'hardcore'")
        require_int(self.hardcore_attempts, "hardcore_attempts", minimum=1)
        if not isfinite(float(self.eps)) or float(self.eps) <= 0:
            raise ValueError("eps must be a positive finite float")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.refinement is not None and not isinstance(self.refinement, RefinementSchedule):
            raise TypeError("refinement must be a RefinementSchedule or None")

    def _validate_cross_field(self) -> None:
        if self.sampling == "hardcore":
            if self.sigma is None or not isfinite(float(self.sigma)) or float(self.sigma) <= 0:
                raise ValueError("hardcore sampling requires a positive finite sigma")
        elif self.sigma is not None:
            raise ValueError("sigma is only meaningful for hardcore sampling")

    def build(self, context: InstrumentBuildContext) -> ContinuousCandidateField:
        return ContinuousCandidateField(
            context.store,
            context.module,
            context.rng,
            pool_size=self.pool_size,
            mode=self.mode,
            sampling=self.sampling,
            sigma=self.sigma,
            hardcore_attempts=self.hardcore_attempts,
            eps=self.eps,
            name=self.name,
            refinement=self.refinement,
        )
