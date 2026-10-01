"""Fixed isotropic Gaussian scalar profile."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Profile


class Gaussian(Profile):
    r"""An L2-normalized fixed-width Gaussian profile."""

    def __init__(
        self, sigma: float | Tensor, *, normalize_columns: bool = True
    ) -> None:
        super().__init__()
        if not isinstance(normalize_columns, bool):
            raise TypeError("normalize_columns must be a bool")
        self.normalize_columns = normalize_columns
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
        return chart.center_parameter_dim

    def parameter_dof(self, chart: Chart) -> int:
        return chart.intrinsic_dim

    def initialize(self, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import initialize

        return initialize(self, chart, atoms, mode=mode)

    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import evaluate

        return evaluate(self, chart, p)

    def evaluate_with_precision(
        self,
        chart: Chart,
        p: Tensor,
        precision: Tensor,
    ) -> Tensor:
        """Evaluate normalized columns with scalar or per-atom precision."""
        from torchcst._backends.torch.profiles.gaussian import evaluate_with_precision

        return evaluate_with_precision(self, chart, p, precision)

    def evaluate_with_precision_slice(
        self,
        chart: Chart,
        p: Tensor,
        precision: Tensor,
        selection: slice | Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import (
            evaluate_with_precision_slice,
        )

        return evaluate_with_precision_slice(self, chart, p, precision, selection)

    def extra_repr(self) -> str:
        return (
            f"sigma={self.sigma.item():g}, normalize_columns={self.normalize_columns}"
        )

    def tangent_config(self) -> tuple:
        return () if self.normalize_columns else (False,)

    @property
    def supports_tangent(self) -> bool:
        return True

    def tangent(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.profiles.gaussian import tangent

        return tangent(self, chart, p)

    def tangent_with_precision(self, chart: Chart, p: Tensor, precision: Tensor):
        """Analytic values, center derivatives and precision derivative.

        Includes the complete L2 normalization, with no amplitude division.
        """
        from torchcst._backends.torch.profiles.gaussian import tangent_with_precision

        return tangent_with_precision(self, chart, p, precision)

    def project_gradient(self, chart: Chart, p: Tensor, gradient: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import project_gradient

        return project_gradient(self, chart, p, gradient)

    def apply_parameter_update(
        self,
        chart: Chart,
        p: Tensor,
        displacement: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import apply_parameter_update

        return apply_parameter_update(self, chart, p, displacement)

    def transport_state(
        self,
        chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.gaussian import transport_state

        return transport_state(self, chart, old, new, state)
