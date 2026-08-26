"""One pullback Adam for every CST parameter -- one class, one template.

``CSTPullbackAdam`` owns the *CST half* of a model -- every site's ``w``,
``s``, ``t``, every learnable chart's ``mu``, every factor bandwidth
``sigma`` -- and deliberately nothing dense: pair it with an ordinary AdamW
for attention and friends.  It is a single optimizer, not a coordinator of
optimizers: all state lives here and every block steps through the same
template,

    r = G^-1/2 g;   Adam moments on r;   delta = -eta * G^-1/2 Adam(r),

capped in the block's own units.  The blocks differ only in what ``G`` is:

======  ==============================================================
block   G = J^T J restricted to it
======  ==============================================================
w       the cross-atom Gram ``G = L L^T`` of the gauge-delivered
        columns (snapshot at construction, :meth:`refresh` after
        coordinate drift) -- ``L^-1`` plays ``G^-1/2``
s, t    the per-atom diagonal coordinate metric (the PullbackAdam
        ``diag`` closed form, reused verbatim)
mu      the per-neuron diagonal chart metric (the ChartPullbackAdam
        closed form)
sigma   the bandwidth metric ``||dW/dsigma||^2`` (Gaussian closed
        form, projected for the unit-column gauge)
======  ==============================================================

Coordinates, charts and bandwidths recompute their ``G`` from the live
geometry every step; the amplitude Gram alone is a snapshot because its
Cholesky is O(K^3) -- the same moving-frame concession PullbackAdam's
``tangent`` moments already make explicit.

Dials are uniform: ``target_step`` calibrates each geometric block's first
step in its natural unit (``sigma`` for coordinates and charts, ``sigma``
itself for bandwidths) and ``cap`` bounds every later one.  Amplitudes have
no such natural unit; they take ``lr_w`` directly, exactly as
:class:`PullbackAdam`'s own amplitude seat does, but stepped in the Gram
frame rather than the Euclidean one.

Deliberately unsupported, with explicit errors: gauges other than
:class:`~torchcst.representation.gauges.L2NormalizedColumns` (the template's
within-atom diagonality is that gauge's algebra); structural events (store
versions are pinned; :meth:`refresh` re-fits after coordinate drift only);
non-Gaussian families under the sigma block (pass ``sigma_block=False``);
and the ``block``/``full`` coordinate metrics (the diag tier first -- it is
the form every lm1 result was won with).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..representation import L2NormalizedColumns
from ..storage import SynapseStore
from . import metric as metric_mod
from .pullback import _metric_block
from .sigma import _bandwidth_jacobian_sq
from .wbasis import _live_gram

__all__ = ["CSTPullbackAdam"]


def _cst_sites(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Every module that *is* a CST site, gauge unchecked.

    Broader than :func:`~torchcst.optim.coordinator.is_continuous_site` on
    purpose: that predicate silently ignores sites in the wrong gauge, and
    the single optimizer must refuse them loudly instead of skipping the
    CST parameters it exists to own.  Wrappers that merely delegate another
    module's store carry no factors and are excluded; a shared store keeps
    its first owner.
    """
    seen: set[int] = set()
    sites: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        store = getattr(module, "synapses", None)
        if not isinstance(store, SynapseStore):
            continue
        if getattr(module, "factor_in", None) is None \
                or getattr(module, "factor_out", None) is None:
            continue
        if id(store) in seen:
            continue
        seen.add(id(store))
        sites.append((name, module))
    return sites


def _adam_direction(
    state: dict[str, Any],
    incoming_parts: list[Tensor],
    betas: tuple[float, float],
    eps: float,
) -> list[Tensor]:
    """The shared template core: tangent moments in, direction out."""
    beta1, beta2 = betas
    state["step"] += 1
    correction1 = 1.0 - beta1 ** state["step"]
    correction2 = 1.0 - beta2 ** state["step"]
    directions = []
    for index, incoming in enumerate(incoming_parts):
        m, v = state["m"][index], state["v"][index]
        m.mul_(beta1).add_(incoming, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(incoming, incoming, value=1.0 - beta2)
        directions.append(
            (m / correction1) / ((v / correction2).sqrt() + eps)
        )
    return directions


class CSTPullbackAdam:
    """The single pullback optimizer for the CST parameters of ``model``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_step: float = 0.01,
        cap: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        damping: float = 1e-2,
        lr_w: float = 1e-3,
        weight_decay_w: float = 0.0,
        w_ridge: float = 1e-2,
        sigma_block: bool = True,
    ) -> None:
        self.model = model
        self._target_step = target_step
        self._cap = cap
        self._betas = betas
        self._eps = eps
        self._damping = damping
        self._lr_w = lr_w
        self._weight_decay_w = weight_decay_w
        self._w_ridge = w_ridge

        self._sites: list[dict[str, Any]] = []
        chart_incidences: dict[int, dict[str, Any]] = {}
        sigma_blocks: dict[int, dict[str, Any]] = {}
        for name, module in _cst_sites(model):
            if not isinstance(module.gauge, L2NormalizedColumns):
                raise TypeError(
                    f"site {name!r}: CSTPullbackAdam is the L2-normalised-"
                    "gauge optimizer (the template's within-atom "
                    "diagonality is that gauge's algebra)"
                )
            store = module.synapses
            chol, live = _live_gram(module, w_ridge)
            lower = chol.to(store.w.dtype).to(store.w.device)
            self._sites.append({
                "name": name,
                "module": module,
                "store": store,
                "live": live.to(store.w.device),
                "lower": lower,
                "version": store.version,
                "coord": {
                    "step": 0, "eta": None,
                    "m": [torch.zeros_like(store.s[live]),
                          torch.zeros_like(store.t[live])],
                    "v": [torch.zeros_like(store.s[live]),
                          torch.zeros_like(store.t[live])],
                },
                "w": {
                    "step": 0,
                    "m": [torch.zeros_like(store.w[live])],
                    "v": [torch.zeros_like(store.w[live])],
                },
            })
            for chart_store in (module.in_neurons, module.out_neurons):
                mu = chart_store.mu
                if isinstance(mu, nn.Parameter) and mu.requires_grad:
                    entry = chart_incidences.setdefault(
                        id(chart_store),
                        {"store": chart_store, "users": [], "step": 0,
                         "eta": None, "m": [torch.zeros_like(mu)],
                         "v": [torch.zeros_like(mu)]},
                    )
                    entry["users"].append(module)
            if sigma_block:
                for side in ("in", "out"):
                    factor = (module.factor_in if side == "in"
                              else module.factor_out)
                    if getattr(factor, "family", None) != "gaussian":
                        raise ValueError(
                            f"site {name!r}: the sigma block has a closed "
                            "form for the gaussian family only; pass "
                            "sigma_block=False to leave bandwidths out"
                        )
                    sigma = factor.sigma
                    if not isinstance(sigma, nn.Parameter) \
                            or not sigma.requires_grad:
                        continue
                    scalar = torch.zeros_like(sigma.detach().sum())
                    entry = sigma_blocks.setdefault(
                        id(sigma),
                        {"sigma": sigma, "users": [], "step": 0,
                         "eta": None, "m": [scalar.clone()],
                         "v": [scalar.clone()]},
                    )
                    entry["users"].append((module, side))
        self._charts = list(chart_incidences.values())
        self._sigmas = list(sigma_blocks.values())

    # -- surface -----------------------------------------------------------

    def parameters(self) -> list[nn.Parameter]:
        """Every parameter this optimizer owns (for exclusion lists)."""
        owned: list[nn.Parameter] = []
        for site in self._sites:
            store = site["store"]
            owned.extend([store.w, store.s, store.t])
        owned.extend(chart["store"].mu for chart in self._charts)
        owned.extend(block["sigma"] for block in self._sigmas)
        return owned

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self, lr_scale: float = 1.0) -> None:
        """Amplitudes, coordinates, charts, then bandwidths."""
        for site in self._sites:
            if site["store"].version != site["version"]:
                raise RuntimeError(
                    f"{site['store'].site!r}: built against store version "
                    f"{site['version']}, now {site['store'].version}; "
                    "structural events are unsupported -- refresh() only "
                    "re-fits after coordinate drift"
                )
            self._step_amplitudes(site, lr_scale)
        for site in self._sites:
            self._step_coordinates(site, lr_scale)
        for chart in self._charts:
            self._step_chart(chart, lr_scale)
        for block in self._sigmas:
            self._step_sigma(block, lr_scale)

    @torch.no_grad()
    def refresh(self) -> None:
        """Re-fit every amplitude Gram after coordinate drift.

        Amplitude moments reset (the follower's row-reset semantics applied
        to a change of frame); every other block recomputes its metric each
        step and needs nothing here.
        """
        for site in self._sites:
            chol, live = _live_gram(site["module"], self._w_ridge)
            store = site["store"]
            site["lower"] = chol.to(store.w.dtype).to(store.w.device)
            site["live"] = live.to(store.w.device)
            site["version"] = store.version
            site["w"] = {
                "step": 0,
                "m": [torch.zeros_like(store.w[site["live"]])],
                "v": [torch.zeros_like(store.w[site["live"]])],
            }

    # -- the four blocks, one template ------------------------------------

    @torch.no_grad()
    def _step_amplitudes(self, site: dict[str, Any], lr_scale: float) -> None:
        store = site["store"]
        grad = store.w.grad
        if grad is None:
            return
        lower, live = site["lower"], site["live"]
        incoming = torch.linalg.solve_triangular(
            lower, grad.index_select(0, live).unsqueeze(1), upper=False
        ).squeeze(1)
        (direction,) = _adam_direction(
            site["w"], [incoming], self._betas, self._eps
        )
        delta = -self._lr_w * lr_scale * direction
        if self._weight_decay_w:
            live_w = store.w.detach().index_select(0, live)
            c = (lower.T @ live_w.unsqueeze(1)).squeeze(1)
            delta = delta - (self._lr_w * self._weight_decay_w) * c
        store.w.index_add_(
            0, live,
            torch.linalg.solve_triangular(
                lower.T, delta.unsqueeze(1), upper=True
            ).squeeze(1),
        )

    @torch.no_grad()
    def _step_coordinates(self, site: dict[str, Any], lr_scale: float) -> None:
        store = site["store"]
        if store.s.grad is None or store.t.grad is None:
            return
        module, live = site["module"], site["live"]
        mu_in = module.in_neurons.mu.detach().to(store.s)
        mu_out = module.out_neurons.mu.detach().to(store.t)
        source = store.s.detach().index_select(0, live)
        target = store.t.detach().index_select(0, live)
        mass_sq = store.w.detach().index_select(0, live).square()
        diag_s, diag_t = _metric_block(
            module, mu_in, mu_out, source, target, mass_sq, compiled=False
        )
        reference = torch.cat(
            [diag_s.reshape(-1), diag_t.reshape(-1)]
        ).median().clamp_min(torch.finfo(diag_s.dtype).tiny)
        whiten_s = (diag_s + self._damping * reference).sqrt()
        whiten_t = (diag_t + self._damping * reference).sqrt()

        incoming = [
            store.s.grad.index_select(0, live) / whiten_s,
            store.t.grad.index_select(0, live) / whiten_t,
        ]
        raw_s, raw_t = _adam_direction(
            site["coord"], incoming, self._betas, self._eps
        )
        raw_s = raw_s / whiten_s
        raw_t = raw_t / whiten_t

        sigma = module.factor_in.sigma.detach().to(raw_s)
        norm = (raw_s.square().sum(1) + raw_t.square().sum(1)).sqrt()
        state = site["coord"]
        if state["eta"] is None:
            median = norm.median().clamp_min(1e-30)
            state["eta"] = float(self._target_step * sigma / median)
        delta_s = raw_s * (state["eta"] * lr_scale)
        delta_t = raw_t * (state["eta"] * lr_scale)
        delta_norm = (
            delta_s.square().sum(1) + delta_t.square().sum(1)
        ).sqrt()
        scale = (
            (self._cap * sigma) / delta_norm.clamp_min(1e-30)
        ).clamp(max=1.0)
        store.s.index_add_(0, live, -delta_s * scale[:, None])
        store.t.index_add_(0, live, -delta_t * scale[:, None])

    @torch.no_grad()
    def _step_chart(self, chart: dict[str, Any], lr_scale: float) -> None:
        store = chart["store"]
        if store.mu.grad is None:
            return
        mu = store.mu.detach()
        diagonal = torch.zeros_like(mu)
        sigma_floor = None
        for module in chart["users"]:
            for factor, synapses, side in (
                (module.factor_in, module.synapses, "in"),
                (module.factor_out, module.synapses, "out"),
            ):
                anchor = (module.in_neurons if side == "in"
                          else module.out_neurons)
                if anchor is not store:
                    continue
                slots = synapses.live_slots().to(mu.device)
                coords = synapses.s if side == "in" else synapses.t
                centers = coords.detach().index_select(0, slots).to(mu)
                mass_sq = synapses.w.detach().index_select(0, slots) \
                    .to(mu).square()
                unit, slope = metric_mod.normalized_columns(
                    factor, mu, centers
                )
                weight = mass_sq[None, :] * slope.square() * (
                    1.0 - unit.square()
                )
                for axis in range(mu.shape[1]):
                    diff = mu[:, axis][:, None] - centers[:, axis][None, :]
                    diagonal[:, axis] += (weight * diff.square()).sum(1)
                factor_sigma = float(factor.sigma.detach().min())
                sigma_floor = (
                    factor_sigma if sigma_floor is None
                    else min(sigma_floor, factor_sigma)
                )
        reference = diagonal.reshape(-1).median().clamp_min(
            torch.finfo(diagonal.dtype).tiny
        )
        whiten = (diagonal + self._damping * reference).sqrt()
        (raw,) = _adam_direction(
            chart, [store.mu.grad / whiten], self._betas, self._eps
        )
        raw = raw / whiten
        norm = raw.square().sum(1).sqrt()
        if chart["eta"] is None:
            chart["eta"] = float(
                self._target_step * sigma_floor
                / float(norm.median().clamp_min(1e-30))
            )
        delta = raw * (chart["eta"] * lr_scale)
        scale = (
            (self._cap * sigma_floor)
            / delta.square().sum(1).sqrt().clamp_min(1e-30)
        ).clamp(max=1.0)
        store.mu.sub_(delta * scale[:, None])

    @torch.no_grad()
    def _step_sigma(self, block: dict[str, Any], lr_scale: float) -> None:
        sigma: nn.Parameter = block["sigma"]
        if sigma.grad is None:
            return
        metric = sum(
            _bandwidth_jacobian_sq(module, side, sigma)
            for module, side in block["users"]
        )
        gearing = metric.clamp_min(torch.finfo(metric.dtype).tiny).sqrt()
        incoming = (sigma.grad.sum() / gearing).reshape(())
        (direction,) = _adam_direction(
            block, [incoming], self._betas, self._eps
        )
        unit = float(sigma.detach().min())
        if block["eta"] is None:
            scale = float(direction.abs().clamp_min(1e-30))
            block["eta"] = self._target_step * unit / scale
        delta = float(-block["eta"] * lr_scale * direction)
        limit = self._cap * unit
        sigma.add_(max(-limit, min(limit, delta)))

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        def pack(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "step": state["step"],
                "eta": state.get("eta"),
                "m": [tensor.clone() for tensor in state["m"]],
                "v": [tensor.clone() for tensor in state["v"]],
            }

        return {
            "sites": {
                site["name"]: {
                    "coord": pack(site["coord"]),
                    "w": pack(site["w"]),
                    "version": site["version"],
                }
                for site in self._sites
            },
            "charts": [pack(chart) for chart in self._charts],
            "sigmas": [pack(block) for block in self._sigmas],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        def unpack(state: dict[str, Any], saved: Mapping[str, Any]) -> None:
            state["step"] = saved["step"]
            if "eta" in state or saved.get("eta") is not None:
                state["eta"] = saved.get("eta")
            for slot, tensor in zip(state["m"], saved["m"], strict=True):
                slot.copy_(tensor)
            for slot, tensor in zip(state["v"], saved["v"], strict=True):
                slot.copy_(tensor)

        for site in self._sites:
            saved = payload["sites"][site["name"]]
            unpack(site["coord"], saved["coord"])
            unpack(site["w"], saved["w"])
            site["version"] = saved["version"]
        for chart, saved in zip(
            self._charts, payload["charts"], strict=True
        ):
            unpack(chart, saved)
        for block, saved in zip(
            self._sigmas, payload["sigmas"], strict=True
        ):
            unpack(block, saved)

    def __repr__(self) -> str:
        return (
            f"CSTPullbackAdam(sites={len(self._sites)}, "
            f"charts={len(self._charts)}, sigmas={len(self._sigmas)})"
        )
