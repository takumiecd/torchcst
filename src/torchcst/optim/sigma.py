"""The bandwidth block of the full pullback: sigma steps priced in W units.

``sigma`` is the one CST parameter every shipped optimizer stepped blind: it
enters :mod:`torchcst.optim.metric` only as a detached constant, while its
own gradient was left to a Euclidean Adam.  For a *scalar* parameter plain
Adam is nearly scale-invariant (``m/sqrt(v)`` cancels any fixed gearing), so
the value of a pullback here is not the division itself but what
:class:`~torchcst.optim.pullback.PullbackAdam` buys its coordinates: a
first-step calibration and a trust cap expressed in the geometry's own
units.  :class:`~torchcst.optim.cst_pullback_adam.CSTPullbackAdam`'s sigma block mirrors exactly that contract for every
factor bandwidth of a model's continuous sites:

- the metric is the diagonal (per-atom) bandwidth Jacobian
  ``J2 = sum_k w_k^2 * ||d column_k / d sigma||^2 * ||other column_k||^2``,
  recomputed each step from the live columns.  Under
  :class:`~torchcst.representation.gauges.L2NormalizedColumns` the delivered
  column is the unit column, so the derivative is the projected one,
  ``(I - u u^T) (d kappa / d sigma) / ||kappa||`` -- the same projection
  that turns the coordinate closed form into its gauge-exact version;
- moments are tangent-space (``r = g / sqrt(J2 + damping)``), the
  first step is calibrated to ``target_step * sigma``, and every step is
  capped at ``cap * sigma`` -- a bandwidth never jumps a meaningful
  fraction of itself in one update.

Closed forms are Gaussian-only (``d kappa / d sigma = kappa * d^2 /
sigma^3``); any other family raises rather than silently receiving a wrong
metric, the repository-wide contract.  A sigma parameter shared by several
sides or sites accumulates one metric term per use and steps once.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..representation import L2NormalizedColumns

__all__: list[str] = []


def _bandwidth_jacobian_sq(
    site: nn.Module, side: str, sigma: Tensor
) -> Tensor:
    """Diagonal ``||dW/d sigma||^2`` contribution of one side of one site.

    Per-atom (cross-atom overlap ignored -- the same tier as the coordinate
    ``diag`` metric), summed over live atoms, in the site's gauge.
    """
    store = site.synapses
    live = store.live_slots().to(store.s.device)
    if side == "in":
        mu = site.in_neurons.mu.detach()
        centers = store.s.detach().index_select(0, live)
        other = site.out_neurons.mu.detach(), store.t.detach() \
            .index_select(0, live), site.factor_out.sigma.detach()
    else:
        mu = site.out_neurons.mu.detach()
        centers = store.t.detach().index_select(0, live)
        other = site.in_neurons.mu.detach(), store.s.detach() \
            .index_select(0, live), site.factor_in.sigma.detach()
    weights = store.w.detach().index_select(0, live)

    squared = torch.cdist(mu, centers).square()
    sig = sigma.detach()
    kappa = torch.exp(-squared / (2.0 * sig.square()))
    dkappa = kappa * squared / sig.pow(3)
    norm_sq = kappa.square().sum(dim=0)
    d_norm_sq = dkappa.square().sum(dim=0)
    if isinstance(site.gauge, L2NormalizedColumns):
        # Projected, unit-column derivative; the other side's column is unit.
        inner = (kappa * dkappa).sum(dim=0)
        tiny = torch.finfo(norm_sq.dtype).tiny
        column_sq = (
            d_norm_sq - inner.square() / norm_sq.clamp_min(tiny)
        ).clamp_min(0.0) / norm_sq.clamp_min(tiny)
        other_sq = torch.ones_like(column_sq)
    else:
        column_sq = d_norm_sq
        other_mu, other_centers, other_sigma = other
        other_sq = torch.exp(
            -torch.cdist(other_mu, other_centers).square()
            / (2.0 * other_sigma.square())
        ).square().sum(dim=0)
    return (weights.square() * column_sq * other_sq).sum()
