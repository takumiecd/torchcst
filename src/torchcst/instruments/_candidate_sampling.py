"""Candidate-pool sampling for continuous charts (uniform / hardcore / local).

Private support module for :mod:`torchcst.instruments.continuous_candidate`:
everything here is pure sampling over ``Box`` domains -- no scoring, no
instrument state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst.representation import Box


def _coarse_spacing(domain: Box, pool_size: int) -> float:
    """Closed-form expected nearest-neighbour spacing for a uniform pool.

    ``width * pool_size ** (-1/dim)`` is the same ``M^(-1/dim)`` scaling
    Stage 0 measured empirically for the 3D input chart (M candidates,
    dim-3 domain -> spacing scales as ``M^(-1/3)``). Using the closed form
    instead of an actual nearest-neighbour computation on a realized pool
    means computing a refinement radius costs no extra ``cdist``/host-sync
    beyond the sampling that already happens.

    ``domain.width`` rather than ``domain.hi - domain.lo``: the law is
    ``(volume / M) ** (1/dim)``, and ``Box.width`` is exactly
    ``volume ** (1/dim)`` -- identical to ``hi - lo`` on a cube, and defined
    on an anisotropic box, where ``hi - lo`` is not a single number at all.
    """
    if pool_size <= 0:
        return domain.width
    return domain.width * (float(pool_size) ** (-1.0 / domain.dim))


def _sample_local_uniform(
    domain: Box,
    centers: Tensor,
    radius: float,
    samples_per_winner: int,
    rng: torch.Generator,
) -> Tensor:
    """Draw ``samples_per_winner`` points around each row of ``centers``.

    Points are drawn uniformly from the axis-aligned cube of half-width
    ``radius`` centered on each row, then clamped into ``domain`` via
    :meth:`Box.retract` -- the same coordinate clip every candidate already
    goes through at birth, so a winner near the domain boundary gets a
    truncated (not reflected or wrapped) local box.
    """
    k = centers.shape[0]
    if k == 0 or samples_per_winner == 0:
        return centers.new_zeros((0, domain.dim))
    device = getattr(rng, "device", torch.device("cpu"))
    unit = torch.rand(
        (k * samples_per_winner, domain.dim), generator=rng, device=device
    )
    offsets = unit.mul(2.0 * radius).add(-radius)
    anchors = centers.to(device=device, dtype=offsets.dtype).repeat_interleave(
        samples_per_winner, dim=0
    )
    return domain.retract(anchors + offsets)


@dataclass(frozen=True)
class PoolSampler:
    """Candidate sampling for one site: the domains, dtype anchors, and the
    optional hardcore-rejection rule (candidates must clear ``sigma`` from
    every live atom, measured in the joint source×target chart)."""

    domain_in: Box
    domain_out: Box
    rng: torch.Generator
    like_source: Tensor
    like_target: Tensor
    sampling: str
    sigma: float | None
    attempts: int

    def uniform(self, m: int) -> tuple[Tensor, Tensor]:
        """Plain uniform draw of ``m`` (source, target) rows."""
        source = self.domain_in.sample(m, self.rng).to(self.like_source)
        target = self.domain_out.sample(m, self.rng).to(self.like_target)
        return source, target

    def pool(
        self, live_source: Tensor, live_target: Tensor, m: int
    ) -> tuple[Tensor, Tensor]:
        """Draw the coarse pool, honoring the hardcore rule when configured."""
        if self.sampling != "hardcore":
            return self.uniform(m)
        return self._rejection_fill(self.uniform, live_source, live_target, m)

    def local(
        self,
        winners_source: Tensor,
        winners_target: Tensor,
        samples_per_winner: int,
        radius_in: float,
        radius_out: float,
        live_source: Tensor,
        live_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample a refinement pool anchored on each coarse winner.

        Under the hardcore rule every retry re-centers on the same winners
        rather than the whole domain: falling back to whole-domain sampling
        on a hardcore miss would defeat the point of refining locally.
        """
        m = winners_source.shape[0] * samples_per_winner

        def draw(count: int) -> tuple[Tensor, Tensor]:
            del count  # local draws are fixed-size: one box per winner.
            source = _sample_local_uniform(
                self.domain_in, winners_source, radius_in, samples_per_winner, self.rng
            ).to(self.like_source)
            target = _sample_local_uniform(
                self.domain_out, winners_target, radius_out, samples_per_winner, self.rng
            ).to(self.like_target)
            return source, target

        if self.sampling != "hardcore" or m == 0:
            return draw(m)
        if live_source.shape[0] == 0:
            return draw(m)
        return self._rejection_fill(draw, live_source, live_target, m)

    def _rejection_fill(
        self,
        draw: Callable[[int], tuple[Tensor, Tensor]],
        live_source: Tensor,
        live_target: Tensor,
        m: int,
    ) -> tuple[Tensor, Tensor]:
        """Collect ``m`` candidates clearing ``sigma`` from every live atom.

        ``draw(count)`` produces fresh candidate batches (a fixed-size draw
        may ignore ``count``). Sampling stops after ``attempts`` rounds; any
        shortfall is filled from one final unfiltered draw, since a
        nearly-covered domain can make strict rejection sampling fail to
        reach a full pool within a bounded budget.
        """
        assert self.sigma is not None
        live_source = live_source.to(self.like_source)
        live_target = live_target.to(self.like_target)
        k_live = live_source.shape[0]
        kept_source: list[Tensor] = []
        kept_target: list[Tensor] = []
        collected = 0
        for _ in range(self.attempts):
            if collected >= m:
                break
            cand_source, cand_target = draw(m - collected)
            if k_live:
                d_in = torch.cdist(cand_source, live_source)
                d_out = torch.cdist(cand_target, live_target)
                joint = torch.sqrt(d_in.square() + d_out.square())
                keep = joint.amin(dim=1) > self.sigma
            else:
                keep = torch.ones(
                    cand_source.shape[0], dtype=torch.bool, device=cand_source.device
                )
            if bool(keep.any()):
                kept_source.append(cand_source[keep])
                kept_target.append(cand_target[keep])
                collected += int(keep.sum())
        source = (
            torch.cat(kept_source, dim=0)
            if kept_source
            else self.like_source.new_zeros((0, self.domain_in.dim))
        )
        target = (
            torch.cat(kept_target, dim=0)
            if kept_target
            else self.like_target.new_zeros((0, self.domain_out.dim))
        )
        if source.shape[0] >= m:
            return source[:m], target[:m]
        remaining = m - source.shape[0]
        fill_source, fill_target = draw(remaining)
        return (
            torch.cat((source, fill_source[:remaining]), dim=0),
            torch.cat((target, fill_target[:remaining]), dim=0),
        )
