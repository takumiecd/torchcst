"""Pullback optimizer for the sample points of one continuous chart.

A chart's points are shared by every incident site-side, so one neuron's
update must see every map it deforms.  The design note in
``docs/pullback-adam.md`` ("Chart coordinates") fixes the mathematics this
module implements:

- the Jacobian of one chart point stacks into the *product* of the incident
  maps, hence the pullback metric is the **sum of per-incidence pullbacks**;
- cross-*incidence* coupling really is zero (different components of the
  product), but within one side the normalization makes one chart point
  perturb every row, so the cross-neuron block is second order in the column
  entries rather than zero.  The per-neuron block is therefore the same
  deliberate block-Jacobi cut the atom metric makes — an early draft claimed
  exactness by disjointness of rows and the autograd oracle in the tests
  refuted it.  Its within-neuron content is exact per atom, through the
  ``(1 - p_i^2)`` projection;
- chart-atom cross coupling stays neglected, the same one-owner split the
  site optimizer already makes;
- the loss gradient needs no assembly: the chart is one shared parameter
  and autograd delivers ``mu.grad`` summed over incidences.

The metric is evaluated through the factor contract
(:func:`torchcst.optim.metric.normalized_columns`), so any registered radial
family works; nothing here is Gaussian-specific.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from ..representation import L2NormalizedColumns
from ..storage import NeuronStore
from . import metric as metric_mod
from .pullback import _inverse_sqrt

__all__ = ["ChartPullbackAdam", "ChartRepulsion"]


class ChartRepulsion:
    """Gradient of ``mu_r * sum_{i<j} exp(-r_ij^2 / 2)`` on one chart.

    The single-sided specialization of :class:`PairRepulsion`:
    ``r^2 = |mu_i - mu_j|^2 / sigma^2``.  Amplitudes do not exist for chart
    points, so the force is uniform by construction.  Above the ``pairs``
    budget, uniformly sampled ordered pairs give an unbiased stochastic
    gradient through the supplied generator.
    """

    def __init__(self, mu: float, *, pairs: int = 1 << 18) -> None:
        if not isinstance(mu, (int, float)) or mu < 0:
            raise ValueError("mu must be a non-negative number")
        if not isinstance(pairs, int) or pairs < 1:
            raise ValueError("pairs must be a positive int")
        self.mu = float(mu)
        self.pairs = pairs

    def gradient(
        self,
        coords: Tensor,
        live: Tensor,
        sigma: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Full-capacity gradient, zero on dead rows."""
        grad = torch.zeros_like(coords)
        count = live.numel()
        if self.mu == 0.0 or count < 2:
            return grad
        rows = coords.index_select(0, live)

        if count * (count - 1) <= self.pairs:
            diff = (rows[:, None, :] - rows[None, :, :]) / sigma
            weight = self.mu * torch.exp(-0.5 * diff.square().sum(-1))
            weight.fill_diagonal_(0.0)
            local = -(weight[..., None] * diff).sum(1) / sigma
        else:
            draw = torch.randint(
                0, count, (2, self.pairs),
                device=coords.device, generator=generator,
            )
            row, col = draw[0], draw[1]
            keep = row != col
            row, col = row[keep], col[keep]
            diff = (rows[row] - rows[col]) / sigma
            weight = self.mu * torch.exp(-0.5 * diff.square().sum(-1))
            scale = count * (count - 1) / float(row.numel())
            local = torch.zeros_like(rows)
            local.index_add_(0, row, -(weight[:, None] * diff) / sigma)
            local.mul_(scale)
        grad.index_copy_(0, live, local)
        return grad


class ChartPullbackAdam:
    """PullbackAdam for the ``mu`` of one :class:`NeuronStore`.

    ``sites`` are the incident continuous CST maps; each contributes one
    pullback term per side that reads this chart.  Everything downstream of
    the metric mirrors :class:`PullbackAdam`: tangent or parameter moments,
    first-step ``target_step`` calibration against the chart's factor sigma,
    a per-neuron cap, accumulated travel, and an optional wall.  Charts have
    no lifecycle, so there is no follower contract here; dormant rows are
    masked out and never move.
    """

    def __init__(
        self,
        store: NeuronStore,
        sites: Iterable[nn.Module],
        *,
        moment_space: str = "tangent",
        metric: str = "diag",
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        cap_sigma: float = 0.1,
        target_step: float = 0.01,
        damping: float = 1e-2,
        repulsion: ChartRepulsion | None = None,
        wall: bool = False,
        seed: int = 0,
    ) -> None:
        if not isinstance(store, NeuronStore):
            raise TypeError("store must be a NeuronStore")
        if not isinstance(store.mu, nn.Parameter):
            raise TypeError(
                "ChartPullbackAdam requires a learnable chart (mu must be a "
                "Parameter; index charts stay buffers)"
            )
        sites = list(sites)
        if not sites:
            raise ValueError("at least one incident site is required")
        if moment_space not in ("parameter", "tangent"):
            raise ValueError("moment_space must be 'parameter' or 'tangent'")
        if metric not in ("diag", "block"):
            raise ValueError("metric must be 'diag' or 'block'")
        for value, name in (
            (cap_sigma, "cap_sigma"),
            (target_step, "target_step"),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        for value, name in ((eps, "eps"), (damping, "damping")):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if repulsion is not None and not isinstance(repulsion, ChartRepulsion):
            raise TypeError("repulsion must be a ChartRepulsion")
        if not isinstance(wall, bool):
            raise TypeError("wall must be a bool")

        # (factor, synapse store, side) per incidence; a self-map site that
        # reads the chart on both sides contributes two incidences.
        incidences = []
        boxes = []
        for site in sites:
            if not isinstance(getattr(site, "gauge", None), L2NormalizedColumns):
                raise TypeError(
                    "ChartPullbackAdam requires L2NormalizedColumns sites"
                )
            hit = False
            if getattr(site, "in_neurons", None) is store:
                incidences.append((site.factor_in, site.synapses, "in"))
                boxes.append(site.synapses.spec.domain_in)
                hit = True
            if getattr(site, "out_neurons", None) is store:
                incidences.append((site.factor_out, site.synapses, "out"))
                boxes.append(site.synapses.spec.domain_out)
                hit = True
            if not hit:
                raise ValueError(
                    f"site {site!r} does not read chart {store.site!r}"
                )
        reference_sigma = float(incidences[0][0].sigma.detach())
        for factor, _, _ in incidences[1:]:
            if abs(float(factor.sigma.detach()) - reference_sigma) > 1e-9:
                raise ValueError(
                    "incident factors disagree on sigma; one chart has one "
                    "scale"
                )
        for box in boxes[1:]:
            if (
                tuple(box.lo_per_axis) != tuple(boxes[0].lo_per_axis)
                or tuple(box.hi_per_axis) != tuple(boxes[0].hi_per_axis)
            ):
                raise ValueError(
                    "incident domains disagree on the chart box"
                )

        self.store = store
        self.incidences = incidences
        self.box = boxes[0]
        self.moment_space = moment_space
        self.metric = metric
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.cap_sigma = float(cap_sigma)
        self.target_step = float(target_step)
        self.damping = float(damping)
        self.repulsion = repulsion
        self.wall = wall
        self.seed = int(seed)
        self.eta: float | None = None
        self.step_count = 0
        self.m = torch.zeros_like(store.mu)
        self.v = torch.zeros_like(store.mu)
        self.travel = torch.zeros(
            store.mu.shape[0], device=store.mu.device, dtype=store.mu.dtype
        )
        self._generator: torch.Generator | None = None

    # ---- metric ------------------------------------------------------------

    def _sigma(self) -> Tensor:
        return self.incidences[0][0].sigma.detach().to(self.store.mu)

    def _pair_generator(self) -> torch.Generator:
        device = self.store.mu.device
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def _metric_blocks(self, live: Tensor) -> Tensor:
        """``[n_live, d, d]`` sum of per-incidence pullback blocks.

        Per incidence, with ``unit``/``slope`` from the factor contract and
        the chart on the side whose atom coordinates are ``c``:

        ``G_ab(mu_i) = sum_k w_k^2 slope_ik^2 (1 - unit_ik^2)
        (mu_ia - c_ka)(mu_ib - c_kb)`` — the projected derivative of the
        delivered L2-normalized column (single-entry perturbation, so the
        tangent projection is the exact ``1 - unit^2`` factor).
        """
        mu_rows = self.store.mu.detach().index_select(0, live)
        d = mu_rows.shape[1]
        blocks = torch.zeros(
            mu_rows.shape[0], d, d,
            device=mu_rows.device, dtype=mu_rows.dtype,
        )
        for factor, synapses, side in self.incidences:
            slots = synapses.live_slots().to(mu_rows.device)
            coords = synapses.s if side == "in" else synapses.t
            centers = coords.detach().index_select(0, slots).to(mu_rows)
            mass_sq = (
                synapses.w.detach().index_select(0, slots).to(mu_rows).square()
            )
            unit, slope = metric_mod.normalized_columns(
                factor, mu_rows, centers
            )
            weight = mass_sq[None, :] * slope.square() * (
                1.0 - unit.square()
            )
            for a in range(d):
                diff_a = mu_rows[:, a][:, None] - centers[:, a][None, :]
                for b in range(a, d):
                    diff_b = (
                        diff_a if b == a
                        else mu_rows[:, b][:, None] - centers[:, b][None, :]
                    )
                    entry = (weight * diff_a * diff_b).sum(1)
                    blocks[:, a, b] += entry
                    if b != a:
                        blocks[:, b, a] += entry
        return blocks

    def _whitener(self, live: Tensor):
        blocks = self._metric_blocks(live)
        diagonals = blocks.diagonal(dim1=-2, dim2=-1)
        tiny = torch.finfo(diagonals.dtype).tiny
        reference = diagonals.reshape(-1).median().clamp_min(tiny)
        if self.metric == "diag":
            effective = diagonals + self.damping * reference
            return lambda values: values / effective.sqrt()
        eye = torch.eye(
            blocks.shape[-1], device=blocks.device, dtype=blocks.dtype
        )
        inverse = _inverse_sqrt(blocks + (self.damping * reference) * eye)
        return lambda values: torch.einsum("kab,kb->ka", inverse, values)

    # ---- step --------------------------------------------------------------

    @torch.no_grad()
    def step(self, lr_scale: float = 1.0) -> None:
        """Consume ``mu.grad`` and apply one chart update."""
        if not isinstance(lr_scale, (int, float)) or lr_scale < 0:
            raise ValueError("lr_scale must be a non-negative number")
        store = self.store
        if store.mu.grad is None:
            return
        live = store.live_ids().to(store.mu.device)
        if live.numel() == 0:
            return

        incoming = store.mu.grad.index_select(0, live)
        if self.repulsion is not None:
            force = self.repulsion.gradient(
                store.mu.detach(), live, self._sigma(), self._pair_generator()
            )
            incoming = incoming + force.index_select(0, live)

        whiten = self._whitener(live)
        if self.moment_space == "tangent":
            incoming = whiten(incoming)

        self.step_count += 1
        m_rows = self.m.index_select(0, live)
        v_rows = self.v.index_select(0, live)
        m_rows.mul_(self.beta1).add_(incoming, alpha=1.0 - self.beta1)
        v_rows.mul_(self.beta2).addcmul_(
            incoming, incoming, value=1.0 - self.beta2
        )
        self.m.index_copy_(0, live, m_rows)
        self.v.index_copy_(0, live, v_rows)
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        direction = (m_rows / correction1) / (
            (v_rows / correction2).sqrt() + self.eps
        )
        raw = whiten(direction)

        raw_norm = raw.square().sum(1).sqrt()
        sigma = self._sigma()
        if self.eta is None:
            median = raw_norm.median().clamp_min(1e-30)
            self.eta = float(self.target_step * sigma / median)

        delta = raw * (self.eta * float(lr_scale))
        delta_norm = delta.square().sum(1).sqrt()
        cap = self.cap_sigma * sigma
        scale = (cap / delta_norm.clamp_min(1e-30)).clamp(max=1.0)
        delta = delta * scale[:, None]

        rows = store.mu.detach().index_select(0, live) - delta
        store.mu.detach().index_copy_(0, live, rows)
        travel_rows = self.travel.index_select(0, live)
        travel_rows += delta.square().sum(1).sqrt() / sigma
        self.travel.index_copy_(0, live, travel_rows)

        if self.wall:
            clamped = store.mu.detach().index_select(0, live).clamp(
                min=store.mu.new_tensor(self.box.lo_per_axis),
                max=store.mu.new_tensor(self.box.hi_per_axis),
            )
            store.mu.detach().index_copy_(0, live, clamped)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear the chart gradient this optimizer consumes."""
        if self.store.mu.grad is None:
            return
        if set_to_none:
            self.store.mu.grad = None
        else:
            self.store.mu.grad.zero_()

    # ---- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "moment_space": self.moment_space,
            "metric": self.metric,
            "m": self.m,
            "v": self.v,
            "travel": self.travel,
            "eta": self.eta,
            "step_count": self.step_count,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["moment_space"] != self.moment_space:
            raise ValueError(
                "state was recorded under a different moment space"
            )
        if state["metric"] != self.metric:
            raise ValueError("state was recorded under a different metric")
        self.m.copy_(state["m"])
        self.v.copy_(state["v"])
        self.travel.copy_(state["travel"])
        self.eta = state["eta"]
        self.step_count = int(state["step_count"])
        self.seed = int(state["seed"])
