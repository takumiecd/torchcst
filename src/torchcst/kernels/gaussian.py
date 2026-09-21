"""Fixed isotropic Gaussian scalar profile."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Profile


class Gaussian(Profile):
    r"""An L2-normalized fixed-width Gaussian profile."""

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
        return chart.embedding_dim

    def parameter_dof(self, chart: Chart) -> int:
        return chart.intrinsic_dim

    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        return chart.initialize_centers(atoms, mode=mode)

    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        precision = self.sigma.reciprocal().square()
        return self.evaluate_with_precision(chart, p, precision)

    def evaluate_with_precision(
        self,
        chart: Chart,
        p: Tensor,
        precision: Tensor,
    ) -> Tensor:
        """Evaluate normalized columns with scalar or per-atom precision."""

        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        if precision.ndim not in (0, 1):
            raise ValueError("precision must be scalar or have shape [atoms]")
        if precision.ndim == 1 and precision.shape != (p.shape[0],):
            raise ValueError("precision must be scalar or have shape [atoms]")
        precision = precision.to(device=p.device, dtype=p.dtype)
        squared_distance = chart.squared_distance(p)
        log_squared_values = -squared_distance * precision
        return torch.exp(0.5 * torch.log_softmax(log_squared_values, dim=0))

    def extra_repr(self) -> str:
        return f"sigma={self.sigma.item():g}"

    @property
    def supports_tangent(self) -> bool:
        return True

    def tangent(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        values, centers, _ = self.tangent_with_precision(
            chart, p, self.sigma.reciprocal().square()
        )
        return values, centers

    def tangent_with_precision(self, chart: Chart, p: Tensor, precision: Tensor):
        """Analytic values, center derivatives and precision derivative.

        Includes the complete L2 normalization, with no amplitude division.
        """
        precision = precision.to(p)
        values = self.evaluate_with_precision(chart, p, precision)
        offset = chart.center_offsets(p)
        squared = chart.squared_distance(p)
        probability = values.square()
        center_log = 2 * precision.reshape(1, -1, 1) * offset
        center_mean = (probability[..., None] * center_log).sum(0, keepdim=True)
        centers = 0.5 * values[..., None] * (center_log - center_mean)
        precision_mean = (probability * squared).sum(0, keepdim=True)
        widths = 0.5 * values * (precision_mean - squared)
        return values, centers, widths

    def project_gradient(self, chart: Chart, p: Tensor, gradient: Tensor) -> Tensor:
        return chart.geometry.project_tangent(p, gradient)

    def apply_parameter_update(
        self,
        chart: Chart,
        p: Tensor,
        displacement: Tensor,
    ) -> Tensor:
        return chart.geometry.retract(p, displacement)

    def transport_state(
        self,
        chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        return chart.geometry.transport(old, new, state)
