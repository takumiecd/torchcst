"""Policy-tree rules for the fast-construction family.

All higher-order and solve details stay family-private.  The rules consume
finalized tangent evidence, construct data-only storage operations, and never
read an objective; realized acceptance remains the root profit court's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Any

import torch
from torch import Tensor

from torchcst._validation import require_int, require_real
from torchcst.instruments.base import KernelPortRequest
from torchcst.instruments.tangent import TangentSnapshot, TangentStatisticsRequest
from torchcst.representation import Box
from torchcst.storage import SynapseBirth, SynapseRefit, SynapseView

from .proposers import _continuous_lineages
from .registry import RetiredCandidateRegistry

__all__ = ["TangentBirth", "TangentRefit"]


def _relative_ridge(gram: Tensor, ridge: float) -> Tensor:
    if gram.numel() == 0:
        return gram
    scale = torch.diagonal(gram).mean().clamp_min(torch.finfo(gram.dtype).tiny)
    return ridge * scale * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )


def _solve(system: Tensor, rhs: Tensor) -> Tensor:
    try:
        result = torch.linalg.solve(system, rhs)
        if bool(torch.isfinite(result).all()):
            return result
    except torch.linalg.LinAlgError:
        pass
    return torch.linalg.pinv(system) @ rhs


def _columns(port: Any, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    k_in, k_out = port.columns(source, target)
    return k_in.to(source), k_out.to(target)


def _gram_rhs(
    port: Any,
    source: Tensor,
    target: Tensor,
    evidence: TangentSnapshot,
) -> tuple[Tensor, Tensor]:
    k_in, k_out = _columns(port, source, target)
    covariance = evidence.covariance.to(k_in)
    cross = evidence.cross.to(device=k_out.device, dtype=k_out.dtype)
    gram = (k_out.transpose(0, 1) @ k_out) * (
        k_in.transpose(0, 1) @ covariance @ k_in
    )
    rhs = torch.einsum("ok,oi,ik->k", k_out, cross, k_in)
    return gram, rhs, k_in, k_out


def _profile(
    port: Any,
    source: Tensor,
    target: Tensor,
    evidence: TangentSnapshot,
) -> tuple[Tensor, Tensor]:
    gram, rhs, _, _ = _gram_rhs(port, source, target, evidence)
    return rhs[0], gram[0, 0].clamp_min(torch.finfo(gram.dtype).tiny)


def _candidate_profiles(
    port: Any,
    source: Tensor,
    target: Tensor,
    evidence: TangentSnapshot,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return paired-candidate ``(g, a)`` without materializing a pool Gram.

    Candidate search only needs the diagonal profile of each paired
    ``(source[j], target[j])``.  Forming the full ``pool_size²`` Gram is both
    unnecessary and prohibitive at FC-1's default pool size.
    """
    k_in, k_out = _columns(port, source, target)
    covariance = evidence.covariance.to(k_in)
    cross = evidence.cross.to(device=k_out.device, dtype=k_out.dtype)
    rhs = torch.einsum("ok,oi,ik->k", k_out, cross, k_in)
    input_curvature = (k_in * (covariance @ k_in)).sum(dim=0)
    output_curvature = k_out.square().sum(dim=0)
    diagonal = (input_curvature * output_curvature).clamp_min(
        torch.finfo(k_in.dtype).tiny
    )
    return rhs, diagonal


def _gain(rhs: Tensor, step: Tensor, gram: Tensor) -> float:
    value = rhs @ step - 0.5 * step @ gram @ step
    result = float(value.detach().cpu())
    return result if isfinite(result) else float("-inf")


@dataclass
class _EvidenceState:
    request: TangentStatisticsRequest
    instruments: dict[str, Any] = field(default_factory=dict)
    ports: dict[str, Any] = field(default_factory=dict)
    cached: dict[str, tuple[int, TangentSnapshot]] = field(default_factory=dict)

    @property
    def requires(self) -> tuple[Any, ...]:
        return (self.request, KernelPortRequest())

    def bind(self, site: str, instruments: dict[str, Any]) -> None:
        try:
            tangent = instruments[self.request.name]
            kernel = instruments["kernel_port"]
        except KeyError as exc:
            raise ValueError("fast construction requires tangent and kernel ports") from exc
        if not callable(getattr(tangent, "snapshot", None)):
            raise TypeError("tangent instrument must provide snapshot()")
        port = getattr(kernel, "port", None)
        if port is None or not callable(getattr(port, "columns", None)):
            raise TypeError("kernel-port instrument must provide port.columns()")
        self.instruments[site] = tangent
        self.ports[site] = port

    def read(self, view: SynapseView) -> tuple[TangentSnapshot, Any]:
        try:
            instrument = self.instruments[view.site]
            port = self.ports[view.site]
        except KeyError as exc:
            raise RuntimeError(f"fast-construction instruments are not bound for {view.site!r}") from exc
        current = instrument.snapshot()
        cached = self.cached.get(view.site)
        if (
            cached is not None
            and cached[0] == view.version
            and current.weighted_batches == 0.0
        ):
            return cached[1], port
        self.cached[view.site] = (view.version, current)
        return current, port

    def consume(self, site: str, *, keep_cache: bool) -> None:
        self.instruments[site].reset()
        if not keep_cache:
            self.cached.pop(site, None)


@dataclass
class TangentRefit:
    """Event-time tangent Gram backfit emitted as one :class:`SynapseRefit`."""

    state: _EvidenceState
    ridge: float = 1.0e-4
    rent: float | None = None
    every_births: int | str | None = 1
    start_after_events: int = 0
    position_iters: int = 0
    trust: float = 0.01
    consume: bool = True
    _last_refit_count: dict[str, int] = field(default_factory=dict, init=False)
    _event_counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.ridge = require_real(self.ridge, "ridge", nonnegative=True)
        if self.rent is not None:
            self.rent = require_real(self.rent, "rent", nonnegative=True)
        if self.every_births is not None and self.every_births != "K/10":
            require_int(self.every_births, "every_births", minimum=1)
        require_int(self.start_after_events, "start_after_events", minimum=0)
        require_int(self.position_iters, "position_iters", minimum=0)
        self.trust = require_real(self.trust, "trust", positive=True)

    @property
    def requires(self) -> tuple[Any, ...]:
        return self.state.requires

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        self.state.bind(site, instruments)

    def _period(self, view: SynapseView) -> int:
        if self.every_births is None:
            return 0
        if self.every_births != "K/10":
            return int(self.every_births)
        return max(1, ceil(max(int(view.ids.numel()), 1) / 10))

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseRefit, ...]:
        del registry, rng
        require_int(budget, "budget", minimum=0)
        event_count = self._event_counts.get(view.site, 0) + 1
        self._event_counts[view.site] = event_count
        if event_count <= self.start_after_events:
            return ()
        if budget == 0:
            return ()
        count = int(view.ids.numel())
        if count == 0:
            return ()
        if self.every_births is not None:
            last = self._last_refit_count.get(view.site, 0)
            if count - last < self._period(view):
                return ()
        evidence, port = self.state.read(view)
        if self.every_births is not None:
            self._last_refit_count[view.site] = count
        try:
            if evidence.weighted_batches == 0.0:
                return ()
            gram, rhs, k_in, k_out = _gram_rhs(
                port, view.s, view.t, evidence
            )
            delta = _solve(gram + _relative_ridge(gram, self.ridge), rhs)
            if not bool(torch.isfinite(delta).all()):
                return ()
            gain = _gain(rhs, delta, gram)
            if gain <= 0.0 or (self.rent is not None and gain < self.rent):
                return ()
            weights = (view.w.detach() + delta.to(view.w)).detach()
            source = view.s.detach().clone()
            target = view.t.detach().clone()
            if self.position_iters:
                source, target, weights = self._polish_positions(
                    source,
                    target,
                    weights,
                    evidence,
                    port,
                    k_in,
                k_out,
                delta,
                view.domain_in,
                view.domain_out,
            )
            return (
                SynapseRefit(
                    view.site,
                    view.ids.clone(),
                    weights,
                    source if self.position_iters else None,
                    target if self.position_iters else None,
                ),
            )
        finally:
            if self.consume:
                self.state.consume(view.site, keep_cache=False)

    def _polish_positions(
        self,
        source: Tensor,
        target: Tensor,
        weights: Tensor,
        evidence: TangentSnapshot,
        port: Any,
        k_in: Tensor,
        k_out: Tensor,
        delta: Tensor,
        domain_in: Any,
        domain_out: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        covariance = evidence.covariance.to(k_in)
        cross = evidence.cross.to(k_out)
        current = cross.clone()
        for j in range(weights.numel()):
            current -= delta[j].to(current) * torch.outer(
                k_out[:, j], covariance @ k_in[:, j]
            )
        for j in range(weights.numel()):
            current += weights[j].to(current) * torch.outer(
                k_out[:, j], covariance @ k_in[:, j]
            )
            local = TangentSnapshot(
                current,
                covariance,
                evidence.weighted_batches,
                evidence.version,
            )
            source[j], target[j] = _polish(
                port,
                source[j],
                target[j],
                local,
                self.position_iters,
                self.trust,
                domain_in,
                domain_out,
            )
            g, a = _profile(port, source[j : j + 1], target[j : j + 1], local)
            weights[j] = (g / a).to(weights)
            kin_j, kout_j = _columns(
                port, source[j : j + 1], target[j : j + 1]
            )
            current -= weights[j].to(current) * torch.outer(
                kout_j[:, 0], covariance @ kin_j[:, 0]
            )
            k_in[:, j] = kin_j[:, 0]
            k_out[:, j] = kout_j[:, 0]
        return source, target, weights


@dataclass
class TangentBirth:
    """Sequential tangent birth with within-event rank-one deflation."""

    state: _EvidenceState
    pool_size: int = 4096
    multistart: int = 4
    polish_iters: int = 0
    trust: float = 0.01
    rent: float | None = None
    _next_lineage: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        require_int(self.pool_size, "pool_size", minimum=1)
        require_int(self.multistart, "multistart", minimum=1)
        require_int(self.polish_iters, "polish_iters", minimum=0)
        self.trust = require_real(self.trust, "trust", positive=True)
        if self.rent is not None:
            self.rent = require_real(self.rent, "rent", nonnegative=True)

    @property
    def requires(self) -> tuple[Any, ...]:
        return self.state.requires

    def bind_instruments(self, site: str, instruments: dict[str, Any]) -> None:
        self.state.bind(site, instruments)

    def propose(
        self,
        view: SynapseView,
        budget: int,
        registry: RetiredCandidateRegistry,
        rng: torch.Generator,
    ) -> tuple[SynapseBirth, ...]:
        require_int(budget, "budget", minimum=0)
        if budget == 0:
            return ()
        if not isinstance(view.domain_in, Box) or not isinstance(view.domain_out, Box):
            raise TypeError("fast construction requires continuous Box domains")
        evidence, port = self.state.read(view)
        try:
            if evidence.weighted_batches == 0.0:
                return ()
            source_pool = view.domain_in.sample(self.pool_size, rng).to(view.s)
            target_pool = view.domain_out.sample(self.pool_size, rng).to(view.t)
            source: list[Tensor] = []
            target: list[Tensor] = []
            weights: list[Tensor] = []
            residual = TangentSnapshot(
                evidence.cross.clone(),
                evidence.covariance,
                evidence.weighted_batches,
                evidence.version,
            )
            live_scale = (
                float(view.w.abs().max().detach().cpu())
                if view.w.numel()
                else float("inf")
            )
            for _ in range(budget):
                rhs, diag = _candidate_profiles(
                    port, source_pool, target_pool, residual
                )
                gains = rhs.square() / (2.0 * diag)
                order = torch.argsort(gains, descending=True, stable=True)
                best: tuple[float, Tensor, Tensor, Tensor] | None = None
                for index in order[: min(self.multistart, order.numel())]:
                    s = source_pool[index].clone()
                    t = target_pool[index].clone()
                    if self.polish_iters:
                        s, t = _polish(
                            port,
                            s,
                            t,
                            residual,
                            self.polish_iters,
                            self.trust,
                            view.domain_in,
                            view.domain_out,
                        )
                    g, a = _profile(port, s[None, :], t[None, :], residual)
                    gain = float((g.square() / (2.0 * a)).detach().cpu())
                    weight = (g / a).to(view.w)
                    if isfinite(live_scale):
                        weight = weight.clamp(-live_scale, live_scale)
                    if best is None or gain > best[0]:
                        best = (gain, s, t, weight)
                assert best is not None
                gain, s, t, weight = best
                if gain <= 0.0 or (self.rent is not None and gain < self.rent):
                    break
                source.append(s)
                target.append(t)
                weights.append(weight)
                kin, kout = _columns(port, s[None, :], t[None, :])
                covariance = residual.covariance.to(kin)
                residual = TangentSnapshot(
                    residual.cross
                    - weight.to(kout)
                    * torch.outer(kout[:, 0], covariance @ kin[:, 0]),
                    residual.covariance,
                    residual.weighted_batches,
                    residual.version,
                )
            if not weights:
                return ()
            count = len(weights)
            lineages = _continuous_lineages(
                view, count, registry, self._next_lineage
            )
            return (
                SynapseBirth(
                    view.site,
                    torch.stack(source),
                    torch.stack(target),
                    torch.stack(weights),
                    lineages,
                ),
            )
        finally:
            self.state.consume(view.site, keep_cache=False)


def _polish(
    port: Any,
    source: Tensor,
    target: Tensor,
    evidence: TangentSnapshot,
    iterations: int,
    trust: float,
    domain_in: Box,
    domain_out: Box,
) -> tuple[Tensor, Tensor]:
    """Small finite-difference Newton solve with Levenberg damping."""
    theta = torch.cat((source, target)).detach().clone()
    dimension = theta.numel()
    if dimension > 6:
        raise ValueError("fast-construction polish is limited to 6 coordinates")
    low = torch.cat(
        (
            source.new_tensor(domain_in.lo_per_axis),
            target.new_tensor(domain_out.lo_per_axis),
        )
    )
    high = torch.cat(
        (
            source.new_tensor(domain_in.hi_per_axis),
            target.new_tensor(domain_out.hi_per_axis),
        )
    )
    epsilon = max(trust * 0.1, 1.0e-6)

    def objective(point: Tensor) -> float:
        s = point[: source.numel()][None, :]
        t = point[source.numel() :][None, :]
        g, a = _profile(port, s, t, evidence)
        return float((-g.square() / (2.0 * a)).detach().cpu())

    damping = 1.0e-4
    for _ in range(iterations):
        center = objective(theta)
        eye = torch.eye(dimension, dtype=torch.float64)
        gradient = torch.zeros(dimension, dtype=torch.float64)
        hessian = torch.zeros((dimension, dimension), dtype=torch.float64)
        for i in range(dimension):
            plus = theta.clone()
            minus = theta.clone()
            plus[i] = min(float(high[i]), float(theta[i]) + epsilon)
            minus[i] = max(float(low[i]), float(theta[i]) - epsilon)
            span = float(plus[i] - minus[i])
            if span <= 0.0:
                continue
            f_plus, f_minus = objective(plus), objective(minus)
            gradient[i] = (f_plus - f_minus) / span
            half = 0.5 * span
            hessian[i, i] = (f_plus - 2.0 * center + f_minus) / (half * half)
            for j in range(i):
                pp, pm, mp, mm = (theta.clone() for _ in range(4))
                for point, si, sj in (
                    (pp, 1.0, 1.0),
                    (pm, 1.0, -1.0),
                    (mp, -1.0, 1.0),
                    (mm, -1.0, -1.0),
                ):
                    point[i] = (theta[i] + si * epsilon).clamp(low[i], high[i])
                    point[j] = (theta[j] + sj * epsilon).clamp(low[j], high[j])
                mixed = (
                    objective(pp) - objective(pm) - objective(mp) + objective(mm)
                ) / (4.0 * epsilon * epsilon)
                hessian[i, j] = hessian[j, i] = mixed
        accepted = None
        for _ in range(8):
            step = _solve(hessian + damping * eye, -gradient)
            norm = float(torch.linalg.vector_norm(step))
            if norm > trust:
                step *= trust / norm
            candidate = (
                theta.to(torch.float64) + step
            ).clamp(low.to(torch.float64), high.to(torch.float64)).to(theta)
            if objective(candidate) < center:
                accepted = candidate
                damping = max(damping * 0.3, 1.0e-8)
                break
            damping *= 10.0
        if accepted is None:
            break
        theta = accepted
    return theta[: source.numel()], theta[source.numel() :]
