"""Fixed isotropic Gaussian scalar profile."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Profile


class Gaussian(Profile):
    r"""The fixed profile ``exp(-||u-v||² / (2 sigma²))``."""

    def __init__(self, sigma: float | Tensor) -> None:
        super().__init__()
        value = torch.as_tensor(sigma)
        if value.numel() != 1:
            raise ValueError("sigma must be a scalar")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        value = value.detach().clone().reshape(())
        if not torch.isfinite(value) or value <= 0:
            raise ValueError("sigma must be finite and positive")
        self.register_buffer("sigma", value)

    def parameter_dim(self, chart: Chart) -> int:
        return chart.dim

    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        if isinstance(atoms, bool) or not isinstance(atoms, int):
            raise TypeError("atoms must be an integer")
        if atoms < 1:
            raise ValueError("atoms must be positive")
        if mode == "balanced":
            indices = (
                torch.linspace(
                    0,
                    chart.features - 1,
                    atoms,
                    device=chart.coordinates.device,
                )
                .round()
                .to(dtype=torch.long)
            )
            return chart.coordinates.index_select(0, indices)
        if mode != "uniform":
            raise ValueError("mode must be 'balanced' or 'uniform'")

        low = chart.coordinates.amin(dim=0)
        high = chart.coordinates.amax(dim=0)
        unit = torch.rand(
            atoms,
            chart.dim,
            device=chart.coordinates.device,
            dtype=chart.coordinates.dtype,
        )
        return low + unit * (high - low)

    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        squared_distance = (
            (chart.coordinates.unsqueeze(-2) - p.unsqueeze(-3)).square().sum(dim=-1)
        )
        return torch.exp(-squared_distance / (2 * self.sigma.square()))

    def extra_repr(self) -> str:
        return f"sigma={self.sigma.item():g}"
