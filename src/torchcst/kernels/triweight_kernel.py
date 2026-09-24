"""One-chart Triweight operator atoms, evaluated in bounded site blocks."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from torchcst.geometry import Chart
from torchcst.geometry.lazy_chart import StripChart

from .base import AtomInit, Kernel


class TriweightKernel(Kernel):
    r"""Signed amplitude times Triweight of the complete chart distance.

    An atom row is ``[amplitude, center...]`` or, with adaptive width,
    ``[amplitude, center..., log_sigma]``. No input/output factorization is
    assumed. The compact profile is ``max(1 - d²/sigma², 0)^3``.
    """

    def __init__(
        self,
        sigma: float,
        *,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        site_chunk: int = 2048,
        atom_chunk: int = 64,
        checkpoint_blocks: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if (sigma_min is None) != (sigma_max is None):
            raise ValueError("sigma_min and sigma_max must be given together")
        adaptive_width = sigma_min is not None
        if adaptive_width and (
            not math.isfinite(sigma_min)
            or not math.isfinite(sigma_max)
            or sigma_min <= 0
            or not sigma_min <= sigma <= sigma_max
            or sigma_min == sigma_max
        ):
            raise ValueError(
                "require 0 < sigma_min <= sigma <= sigma_max and sigma_min < sigma_max"
            )
        if (
            type(site_chunk) is not int
            or site_chunk < 1
            or type(atom_chunk) is not int
            or atom_chunk < 1
        ):
            raise ValueError("chunk sizes must be positive integers")
        self.register_buffer(
            "sigma", torch.tensor(float(sigma), dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "sigma_min", self.sigma.new_tensor(sigma_min if adaptive_width else sigma)
        )
        self.register_buffer(
            "sigma_max", self.sigma.new_tensor(sigma_max if adaptive_width else sigma)
        )
        self.adaptive_width = adaptive_width
        self.site_chunk = site_chunk
        self.atom_chunk = atom_chunk
        self.checkpoint_blocks = checkpoint_blocks

    def tangent_config(self) -> tuple:
        return (self.adaptive_width,)

    def _check_chart(self, chart: Chart) -> None:
        if not isinstance(chart, Chart) or len(chart.shape) != 2:
            raise TypeError("TriweightKernel requires a two-dimensional Chart")
        if isinstance(chart, StripChart):
            chart.validate_support(float(self.sigma_max))

    def parameter_dim(self, chart: Chart) -> int:
        self._check_chart(chart)
        return 1 + chart.center_parameter_dim + int(self.adaptive_width)

    def parameter_dof(self, chart: Chart) -> int:
        return 1 + chart.intrinsic_dim + int(self.adaptive_width)

    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        self._check_chart(chart)
        centers = chart.initialize_centers(atoms, mode=mode)
        amplitude = centers.new_empty((atoms, 1)).normal_(
            mean=0.0, std=0.1 / math.sqrt(atoms)
        )
        if not self.adaptive_width:
            return torch.cat((amplitude, centers), dim=-1)
        widths = centers.new_full((atoms, 1), float(self.sigma.log()))
        return torch.cat((amplitude, centers, widths), dim=-1)

    def _centers(self, chart: Chart, p: Tensor) -> Tensor:
        return p[:, 1 : 1 + chart.center_parameter_dim]

    def _values(self, chart: Chart, p: Tensor, selection: slice | Tensor) -> Tensor:
        squared = chart.squared_distance(self._centers(chart, p), selection)
        sigma = (
            p[:, -1].clamp(self.sigma_min.log(), self.sigma_max.log()).exp()
            if self.adaptive_width
            else self.sigma
        )
        scaled = squared / sigma.square()
        raw = (1 - scaled).clamp_min(0).pow(3)
        return raw * p[:, 0].unsqueeze(0)

    def _block(self, chart: Chart, p: Tensor, selection: slice | Tensor) -> Tensor:
        parts = []
        for start in range(0, p.shape[0], self.atom_chunk):
            selected = p[start : start + self.atom_chunk]
            if (
                self.checkpoint_blocks
                and torch.is_grad_enabled()
                and selected.requires_grad
            ):
                values = checkpoint(
                    lambda x: self._values(chart, x, selection),
                    selected,
                    use_reentrant=False,
                )
            else:
                values = self._values(chart, selected, selection)
            parts.append(values.sum(dim=-1))
        return torch.stack(parts).sum(dim=0)

    def weight(self, chart: Chart, p: Tensor) -> Tensor:
        """Build only the required dense weight; temporaries are chunk-bounded."""

        self._check_chart(chart)
        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        blocks = [
            self._block(
                chart, p, slice(start, min(start + self.site_chunk, chart.features))
            )
            for start in range(0, chart.features, self.site_chunk)
        ]
        return torch.cat(blocks).reshape(chart.shape)

    def materialize_atoms(self, chart: Chart, p: Tensor) -> Tensor:
        """Explicit diagnostic view [atoms, out, in]; may be large."""

        self._check_chart(chart)
        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        blocks = [
            self._values(
                chart, p, slice(start, min(start + self.site_chunk, chart.features))
            )
            for start in range(0, chart.features, self.site_chunk)
        ]
        return (
            torch.cat(blocks, dim=0).transpose(0, 1).reshape(p.shape[0], *chart.shape)
        )

    def packed_weight(self, chart: StripChart, p: Tensor) -> Tensor:
        """Return contiguous tile-major values [tiles, tile_out, tile_in].

        Edge tiles are zero-padded. A native tile contraction can consume this
        layout directly; no site-ID table is retained with the packed weight.
        """

        if not isinstance(chart, StripChart):
            raise TypeError("packed_weight requires a StripChart")
        self._check_chart(chart)
        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        tiles = []
        tile_size = math.prod(chart.tile_shape)
        for station in range(chart.tile_count):
            logical, local = chart.tile_indices(station)
            values = self._block(chart, p, logical)
            tile = values.new_zeros(tile_size).index_copy(0, local, values)
            tiles.append(tile.reshape(chart.tile_shape))
        return torch.stack(tiles)

    def project_parameter_gradient(
        self, chart: Chart, p: Tensor, gradient: Tensor
    ) -> Tensor:
        if gradient.shape != p.shape:
            raise ValueError("gradient must match p")
        dim = chart.center_parameter_dim
        center = chart.geometry.project_tangent(
            self._centers(chart, p), gradient[:, 1 : 1 + dim]
        )
        parts = (
            (gradient[:, :1], center, gradient[:, -1:])
            if self.adaptive_width
            else (gradient[:, :1], center)
        )
        return torch.cat(parts, dim=-1)

    def apply_parameter_update(
        self, chart: Chart, p: Tensor, displacement: Tensor, *, step_size: float
    ) -> Tensor:
        del step_size
        if displacement.shape != p.shape:
            raise ValueError("displacement must match p")
        dim = chart.center_parameter_dim
        center = chart.geometry.retract(
            self._centers(chart, p), displacement[:, 1 : 1 + dim]
        )
        parts = [p[:, :1] + displacement[:, :1], center]
        if self.adaptive_width:
            width = (p[:, -1:] + displacement[:, -1:]).clamp(
                self.sigma_min.log(), self.sigma_max.log()
            )
            parts.append(width)
        return torch.cat(parts, dim=-1)

    def transport_parameter_state(
        self, chart: Chart, old: Tensor, new: Tensor, state: Tensor
    ) -> Tensor:
        if old.shape != new.shape or old.shape != state.shape:
            raise ValueError("old, new, and state must have matching shapes")
        dim = chart.center_parameter_dim
        center = chart.geometry.transport(
            old[:, 1 : 1 + dim], new[:, 1 : 1 + dim], state[:, 1 : 1 + dim]
        )
        parts = (
            (state[:, :1], center, state[:, -1:])
            if self.adaptive_width
            else (state[:, :1], center)
        )
        return torch.cat(parts, dim=-1)
