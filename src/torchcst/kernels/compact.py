"""Compactly supported isotropic scalar profiles."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart

from .base import AtomInit, Profile


class _CompactRadialProfile(Profile):
    """Compact radial profile with support radius ``sigma``.

    ``evaluate_with_precision`` uses the same convention as ``Gaussian``:
    the support radius is ``R = precision^{-1/2}``. Outside ``r = d/R >= 1``
    the raw value is identically zero. Subclasses may retain the legacy
    discrete column normalization when it is part of their profile semantics.
    """

    normalize_columns = True

    def __init__(
        self,
        sigma: float | Tensor,
        *,
        normalize_columns: bool | None = None,
    ) -> None:
        super().__init__()
        if normalize_columns is not None and not isinstance(normalize_columns, bool):
            raise TypeError("normalize_columns must be a bool or None")
        if normalize_columns is not None:
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
        from torchcst._backends.torch.profiles.compact import initialize

        return initialize(self, chart, atoms, mode=mode)

    def evaluate(self, chart: Chart, p: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import evaluate

        return evaluate(self, chart, p)

    def evaluate_with_precision(
        self,
        chart: Chart,
        p: Tensor,
        precision: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import evaluate_with_precision

        return evaluate_with_precision(self, chart, p, precision)

    def evaluate_with_precision_slice(
        self,
        chart: Chart,
        p: Tensor,
        precision: Tensor,
        selection: slice | Tensor,
    ) -> Tensor:
        """Evaluate raw compact values for a bounded set of operator sites."""
        from torchcst._backends.torch.profiles.compact import (
            evaluate_with_precision_slice,
        )

        return evaluate_with_precision_slice(self, chart, p, precision, selection)

    def extra_repr(self) -> str:
        return (
            f"sigma={self.sigma.item():g}, normalize_columns={self.normalize_columns}"
        )

    def tangent_config(self) -> tuple:
        return (self.normalize_columns,)

    @property
    def supports_tangent(self) -> bool:
        return True

    def tangent(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        from torchcst._backends.torch.profiles.compact import tangent

        return tangent(self, chart, p)

    def tangent_with_precision(self, chart: Chart, p: Tensor, precision: Tensor):
        """Analytic values, center derivatives and precision derivative."""
        from torchcst._backends.torch.profiles.compact import tangent_with_precision

        return tangent_with_precision(self, chart, p, precision)

    def _geometry(
        self, chart: Chart, p: Tensor, precision: Tensor, *, need_offsets: bool = True
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        from torchcst._backends.torch.profiles.compact import _geometry

        return _geometry(self, chart, p, precision, need_offsets=need_offsets)

    def project_gradient(self, chart: Chart, p: Tensor, gradient: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import project_gradient

        return project_gradient(self, chart, p, gradient)

    def apply_parameter_update(
        self,
        chart: Chart,
        p: Tensor,
        displacement: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import apply_parameter_update

        return apply_parameter_update(self, chart, p, displacement)

    def transport_state(
        self,
        chart: Chart,
        old: Tensor,
        new: Tensor,
        state: Tensor,
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import transport_state

        return transport_state(self, chart, old, new, state)

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import _unnormalized_from_squared

        return _unnormalized_from_squared(self, squared, precision)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import _d_raw_d_center

        return _d_raw_d_center(self, offset, squared, precision)

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import _d_raw_d_precision

        return _d_raw_d_precision(self, squared, precision)


class WendlandC2(_CompactRadialProfile):
    r"""The Wendland \(C^2\) profile \((1-r)_+^4(4r+1)\), L2-normalized on the chart."""

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            wendlandc2_unnormalized_from_squared,
        )

        return wendlandc2_unnormalized_from_squared(self, squared, precision)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import wendlandc2_d_raw_d_center

        return wendlandc2_d_raw_d_center(self, offset, squared, precision)

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            wendlandc2_d_raw_d_precision,
        )

        return wendlandc2_d_raw_d_precision(self, squared, precision)


class Triangle(_CompactRadialProfile):
    r"""The raw radial triangle profile :math:`(1-r)_+`.

    This is the lowest-order compact radial profile.  Its center derivative
    uses the finite zero subgradient at ``r = 0`` and is discontinuous at the
    support boundary, so it is intended for first-order optimization only.
    """

    normalize_columns = False

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            triangle_unnormalized_from_squared,
        )

        return triangle_unnormalized_from_squared(self, squared, precision)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import triangle_d_raw_d_center

        return triangle_d_raw_d_center(self, offset, squared, precision)

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import triangle_d_raw_d_precision

        return triangle_d_raw_d_precision(self, squared, precision)


class Biweight(_CompactRadialProfile):
    r"""The raw biweight profile :math:`(1-r^2)_+^2`.

    The value and first derivative vanish continuously at the support
    boundary.  This is the minimum polynomial order in ``r^2`` suited to a
    first-order optimizer without Triangle's boundary-gradient jump.
    """

    normalize_columns = False

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            biweight_unnormalized_from_squared,
        )

        return biweight_unnormalized_from_squared(self, squared, precision)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import biweight_d_raw_d_center

        return biweight_d_raw_d_center(self, offset, squared, precision)

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import biweight_d_raw_d_precision

        return biweight_d_raw_d_precision(self, squared, precision)


class Triweight(_CompactRadialProfile):
    r"""The triweight profile \((1-r^2)_+^3\), L2-normalized by default.

    Set ``normalize_columns=False`` to retain distance and bandwidth
    derivatives when a column is supported by only one site.
    """

    normalize_columns = True

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            triweight_unnormalized_from_squared,
        )

        return triweight_unnormalized_from_squared(self, squared, precision)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        from torchcst._backends.torch.profiles.compact import triweight_d_raw_d_center

        return triweight_d_raw_d_center(self, offset, squared, precision)

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        from torchcst._backends.torch.profiles.compact import (
            triweight_d_raw_d_precision,
        )

        return triweight_d_raw_d_precision(self, squared, precision)
