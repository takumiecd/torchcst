"""Structural forces for the continuous lifecycle: rent and repulsion.

Tensors in, tensors out: these objects are stateless (time arrives as an
argument), know no store, no module and no engine -- the contract
``optim/metric.py`` already keeps.  Both return gradient contributions that
are added *before* Adam's moments.  That placement is load-bearing twice
over: the rent must compete with the loss gradient in magnitude, so that
``|dL/dw| > lam`` prices atoms by how much map they carry (a decay applied
after Adam's normalisation thresholds on signal-to-noise instead and wakes
every consistent atom regardless of size), and the repulsion must still
reach atoms whose loss gradient has vanished with their amplitude.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor


class SmoothRent:
    """``lam(step) * w / sqrt(w^2 + eps^2)`` -- a smooth L1 rent on amplitude.

    ``eps`` keeps a dying atom's amplitude wiggling instead of parking at an
    exact zero: the tiny sign-coherent residue is what turns its coordinate
    gradient into ascent of the squared birth-score field.  ``lam`` may be a
    number or a callable of the step for annealing schedules.
    """

    def __init__(
        self, lam: float | Callable[[int], float], *, eps: float = 1e-3
    ) -> None:
        if callable(lam):
            probe = lam(0)
            if not isinstance(probe, (int, float)) or probe < 0:
                raise ValueError("lam(step) must return a non-negative number")
        elif not isinstance(lam, (int, float)) or lam < 0:
            raise ValueError("lam must be a non-negative number or callable")
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be a positive number")
        self.lam = lam
        self.eps = float(eps)

    def rate(self, step: int) -> float:
        return float(self.lam(step)) if callable(self.lam) else float(self.lam)

    def gradient(self, w: Tensor, step: int) -> Tensor:
        """Rent gradient for the current amplitudes, same shape as ``w``."""
        return self.rate(step) * w / torch.sqrt(w * w + self.eps * self.eps)


class PairRepulsion:
    """Pairwise structural force in joint factor-sigma units.

    ``r^2 = |ds|^2/sigma_in^2 + |dt|^2/sigma_out^2``: two atoms only repel
    when they are close on *both* sides, which is exactly when their columns
    collide.  ``potential="gaussian"`` is the bounded historical energy
    ``mu exp(-r^2/2)``.  ``potential="log_barrier"`` is
    ``-mu log(1-exp(-r^2/2))``: the negative log determinant of the two
    normalized Gaussian columns' Gram matrix.  It vanishes at range and
    diverges as the columns become identical.

    Amplitudes do not enter, so reserves at ``w ~ 0`` feel the same force.
    When the live population exceeds the ``pairs`` budget, uniformly sampled
    ordered pairs give an unbiased stochastic gradient through the supplied
    generator.  Sampling does not itself guarantee that every close pair is
    visited on every step; callers requiring a hard exclusion guarantee need
    an exhaustive or neighbourhood-complete candidate set.
    """

    def __init__(
        self,
        mu: float,
        *,
        pairs: int = 1 << 18,
        potential: Literal["gaussian", "log_barrier"] = "gaussian",
    ) -> None:
        if not isinstance(mu, (int, float)) or mu < 0:
            raise ValueError("mu must be a non-negative number")
        if not isinstance(pairs, int) or pairs < 1:
            raise ValueError("pairs must be a positive int")
        if potential not in ("gaussian", "log_barrier"):
            raise ValueError(
                "potential must be 'gaussian' or 'log_barrier'"
            )
        self.mu = float(mu)
        self.pairs = pairs
        self.potential = potential

    def _weight(self, r_sq: Tensor) -> Tensor:
        overlap_sq = torch.exp(-0.5 * r_sq)
        if self.potential == "gaussian":
            return self.mu * overlap_sq
        # 1-exp(-x) evaluated without cancellation.  The clamp is only a
        # floating-point guard: mathematically the force diverges at x=0.
        denominator = (-torch.expm1(-0.5 * r_sq)).clamp_min(
            torch.finfo(r_sq.dtype).tiny
        )
        return self.mu * overlap_sq / denominator

    def gradient(
        self,
        source: Tensor,
        target: Tensor,
        live: Tensor,
        sigma_in: Tensor,
        sigma_out: Tensor,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """``(g_s, g_t)`` in full-capacity layout, zero on dead rows."""
        g_s = torch.zeros_like(source)
        g_t = torch.zeros_like(target)
        count = live.numel()
        if self.mu == 0.0 or count < 2:
            return g_s, g_t
        s_live = source.index_select(0, live)
        t_live = target.index_select(0, live)

        if count * (count - 1) <= self.pairs:
            ds = (s_live[:, None, :] - s_live[None, :, :]) / sigma_in
            dt = (t_live[:, None, :] - t_live[None, :, :]) / sigma_out
            r_sq = ds.square().sum(-1) + dt.square().sum(-1)
            weight = self._weight(r_sq)
            weight.fill_diagonal_(0.0)
            grad_s = -(weight[..., None] * ds).sum(1) / sigma_in
            grad_t = -(weight[..., None] * dt).sum(1) / sigma_out
        else:
            draw = torch.randint(
                0, count, (2, self.pairs),
                device=source.device, generator=generator,
            )
            row, col = draw[0], draw[1]
            keep = row != col
            row, col = row[keep], col[keep]
            ds = (s_live[row] - s_live[col]) / sigma_in
            dt = (t_live[row] - t_live[col]) / sigma_out
            r_sq = ds.square().sum(-1) + dt.square().sum(-1)
            weight = self._weight(r_sq)
            scale = count * (count - 1) / float(row.numel())
            grad_s = torch.zeros_like(s_live)
            grad_t = torch.zeros_like(t_live)
            grad_s.index_add_(0, row, -scale * weight[:, None] * ds / sigma_in)
            grad_t.index_add_(0, row, -scale * weight[:, None] * dt / sigma_out)

        g_s.index_copy_(0, live, grad_s)
        g_t.index_copy_(0, live, grad_t)
        return g_s, g_t
