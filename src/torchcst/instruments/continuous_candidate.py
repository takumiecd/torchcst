"""Continuous-chart candidate scoring instrument (cRigL/cRES)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import Tensor

from torchcst.compute import ObservationTiming
from torchcst.representation import Box
from torchcst.storage import SynapseStore, SynapseView

from .base import InstrumentBuildContext, KernelPort, WeightedMeasurement, weighted_sum
from .scored import CandidateSnapshot


def _normalize_rows(x: Tensor, eps: float) -> Tensor:
    """Row-normalize ``x``, mapping any near-zero row to the zero vector.

    A near-zero row means the candidate/live atom's kernel column vanishes
    (e.g. a compact-support kernel evaluated far from every neuron); such a
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
    more direct fallback, but it has no MPS kernel (hard error, not just a
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


def _sample_uniform(
    domain_in: Box, domain_out: Box, m: int, rng: torch.Generator, like_source: Tensor, like_target: Tensor
) -> tuple[Tensor, Tensor]:
    source = domain_in.sample(m, rng).to(like_source)
    target = domain_out.sample(m, rng).to(like_target)
    return source, target


def _sample_hardcore(
    domain_in: Box,
    domain_out: Box,
    live_source: Tensor,
    live_target: Tensor,
    m: int,
    sigma: float,
    attempts: int,
    rng: torch.Generator,
    like_source: Tensor,
    like_target: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rejection-sample candidates farther than ``sigma`` from every live atom.

    Distance is Euclidean in the joint (source, target) chart.  Sampling
    stops after ``attempts`` rounds; any shortfall is documented and filled
    with plain uniform draws, since a nearly-covered domain can make strict
    rejection sampling fail to reach a full pool within a bounded budget.
    """
    live_source = live_source.to(like_source)
    live_target = live_target.to(like_target)
    k_live = live_source.shape[0]
    kept_source: list[Tensor] = []
    kept_target: list[Tensor] = []
    collected = 0
    for _ in range(attempts):
        if collected >= m:
            break
        draw = m - collected
        cand_source = domain_in.sample(draw, rng).to(like_source)
        cand_target = domain_out.sample(draw, rng).to(like_target)
        if k_live:
            d_in = torch.cdist(cand_source, live_source)
            d_out = torch.cdist(cand_target, live_target)
            joint = torch.sqrt(d_in.square() + d_out.square())
            keep = joint.amin(dim=1) > sigma
        else:
            keep = torch.ones(draw, dtype=torch.bool, device=cand_source.device)
        if bool(keep.any()):
            kept_source.append(cand_source[keep])
            kept_target.append(cand_target[keep])
            collected += int(keep.sum())
    source = (
        torch.cat(kept_source, dim=0) if kept_source else like_source.new_zeros((0, domain_in.dim))
    )
    target = (
        torch.cat(kept_target, dim=0) if kept_target else like_target.new_zeros((0, domain_out.dim))
    )
    if source.shape[0] >= m:
        return source[:m], target[:m]
    # Budget exhausted (e.g. a nearly-covered domain): documented fallback to
    # plain uniform sampling for the remainder rather than looping forever.
    remaining = m - source.shape[0]
    fill_source, fill_target = _sample_uniform(
        domain_in, domain_out, remaining, rng, like_source, like_target
    )
    return torch.cat((source, fill_source), dim=0), torch.cat((target, fill_target), dim=0)


class ContinuousCandidateField:
    """Score a freshly sampled continuous candidate pool (cRigL/cRES).

    Samples ``pool_size`` fresh ``(source, target)`` candidates from the
    layer's input/output ``Box`` domains whenever the structural version
    changes, accumulates ``G`` (the batch gradient w.r.t. the materialised
    weight, ``[out_features, in_features]``) via the same
    accumulate-until-consumed backward-capture pattern used by
    :class:`~torchcst.instruments.CertificateSubspace`, and scores each
    candidate ``z = (s, t)`` against the direction ``a_z = u(t) v(s)^T``
    (Frobenius-unit-norm, ``u`` from ``kernel_out``/target and ``v`` from
    ``kernel_in``/source -- the same orientation :meth:`dense_weight` uses):

    * ``mode="raw"`` (cRigL): ``score(z) = <G, a_z>``.
    * ``mode="deflated"`` (cRES): the component of ``<G, a_z>`` orthogonal to
      the span of the live atoms' directions, normalized by the residual
      direction norm -- i.e. deflating the score by how much of ``a_z`` is
      already representable by the current live support.

    All kernel evaluation goes through :class:`~torchcst.instruments.base.KernelPort`
    rather than reaching into the compute module's kernel/neuron internals.
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
    ) -> None:
        if not isinstance(store.spec.domain_in, Box) or not isinstance(
            store.spec.domain_out, Box
        ):
            raise TypeError("ContinuousCandidateField requires Box domains")
        if not callable(getattr(module, "kernel_columns", None)):
            raise TypeError("compute module must provide kernel_columns()")
        self.name = name
        self.store = store
        self.module = module
        self.port = KernelPort(module)
        self.rng = rng
        self.pool_size = pool_size
        self.mode = mode
        self.sampling = sampling
        self.sigma = sigma
        self.hardcore_attempts = hardcore_attempts
        self.eps = eps
        self._version = -1
        self._source = store.s.detach().new_zeros((0, store.d_in))
        self._target = store.t.detach().new_zeros((0, store.d_out))
        self._G: Tensor | None = None

    def prepare(self, view: SynapseView, module: Any) -> None:
        """Resample the candidate pool only after the structural version changes."""
        if module is not self.module:
            raise ValueError("instrument was prepared with another compute module")
        if view.version == self._version:
            return
        domain_in = self.store.spec.domain_in
        domain_out = self.store.spec.domain_out
        if self.sampling == "hardcore":
            assert self.sigma is not None
            self._source, self._target = _sample_hardcore(
                domain_in,
                domain_out,
                view.s,
                view.t,
                self.pool_size,
                self.sigma,
                self.hardcore_attempts,
                self.rng,
                self.store.s,
                self.store.t,
            )
        else:
            self._source, self._target = _sample_uniform(
                domain_in, domain_out, self.pool_size, self.rng, self.store.s, self.store.t
            )
        self._version = view.version

    def _measure(self, module: Any, x: Tensor, g_out: Tensor) -> dict[str, Tensor]:
        if module is not self.module:
            raise ValueError("instrument measured another compute module")
        x_flat = x.detach().reshape(-1, x.shape[-1])
        g_flat = g_out.detach().reshape(-1, g_out.shape[-1])
        if x_flat.shape[0] != g_flat.shape[0]:
            raise ValueError("captured x and g_out batch dimensions do not align")
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

        Unlike :class:`~torchcst.instruments.CertificateSubspace`, this
        instrument is not automatically reset by the engine's per-event
        sweep (that sweep only targets ``CertificateSubspace`` instances).
        Callers that want ``G`` cleared between structural events should call
        this explicitly at the appropriate boundary.
        """
        if self._G is not None:
            self._G.zero_()

    def candidate_snapshot(self) -> CandidateSnapshot:
        """Score the current candidate pool against the accumulated ``G``."""
        view = self.store.view()
        live_source = view.s.detach()
        live_target = view.t.detach()
        k_live = live_source.shape[0]

        k_in_pool, k_out_pool = self.port.columns(self._source, self._target)
        u_pool = _normalize_rows(k_out_pool.transpose(0, 1), self.eps)
        v_pool = _normalize_rows(k_in_pool.transpose(0, 1), self.eps)

        if self._G is None:
            gradient = u_pool.new_zeros((self.module.out_features, self.module.in_features))
        else:
            gradient = self._G.detach().to(u_pool)

        raw = (u_pool * (v_pool @ gradient.transpose(0, 1))).sum(dim=1)

        if self.mode == "raw" or k_live == 0:
            scores = raw
        else:
            k_in_live, k_out_live = self.port.columns(live_source, live_target)
            u_live = _normalize_rows(k_out_live.transpose(0, 1), self.eps)
            v_live = _normalize_rows(k_in_live.transpose(0, 1), self.eps)

            b = (u_live * (v_live @ gradient.transpose(0, 1))).sum(dim=1)
            cross = (u_pool @ u_live.transpose(0, 1)) * (v_pool @ v_live.transpose(0, 1))
            gram = (u_live @ u_live.transpose(0, 1)) * (v_live @ v_live.transpose(0, 1))

            coeffs = _solve_gram(gram, cross.transpose(0, 1), self.eps)  # [K, M]
            proj_self = (cross * coeffs.transpose(0, 1)).sum(dim=1)
            proj_grad = coeffs.transpose(0, 1) @ b

            residual = (1.0 - proj_self).clamp_min(0.0)
            valid = residual > self.eps
            safe_residual = residual.clamp_min(self.eps)
            scores = torch.where(
                valid, (raw - proj_grad) / safe_residual.sqrt(), torch.zeros_like(raw)
            )

        return CandidateSnapshot(
            self._source.detach().clone(),
            self._target.detach().clone(),
            scores.detach(),
        )

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

    def __post_init__(self) -> None:
        if isinstance(self.pool_size, bool) or not isinstance(self.pool_size, int):
            raise TypeError("pool_size must be an int")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.mode not in {"raw", "deflated"}:
            raise ValueError("mode must be 'raw' or 'deflated'")
        if self.sampling not in {"uniform", "hardcore"}:
            raise ValueError("sampling must be 'uniform' or 'hardcore'")
        if self.sampling == "hardcore":
            if self.sigma is None or not isfinite(float(self.sigma)) or float(self.sigma) <= 0:
                raise ValueError("hardcore sampling requires a positive finite sigma")
        elif self.sigma is not None:
            raise ValueError("sigma is only meaningful for hardcore sampling")
        if isinstance(self.hardcore_attempts, bool) or not isinstance(
            self.hardcore_attempts, int
        ):
            raise TypeError("hardcore_attempts must be an int")
        if self.hardcore_attempts <= 0:
            raise ValueError("hardcore_attempts must be positive")
        if not isfinite(float(self.eps)) or float(self.eps) <= 0:
            raise ValueError("eps must be a positive finite float")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        try:
            resolved_timing = ObservationTiming(self.timing)
        except ValueError as exc:
            raise ValueError("timing must be backward_inline or after_backward") from exc
        object.__setattr__(self, "sigma", None if self.sigma is None else float(self.sigma))
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "timing", resolved_timing)

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
        )
