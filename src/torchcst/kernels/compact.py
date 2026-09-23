"""Compactly supported isotropic scalar profiles."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst.geometry import Chart
from torchcst.profiling import cst_span

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
        with cst_span("cst.profile.geometry"):
            _, squared, precision = self._geometry(chart, p, precision)
        with cst_span("cst.profile.radial"):
            raw = self._unnormalized_from_squared(squared, precision)
        if not self.normalize_columns:
            return raw
        with cst_span("cst.profile.normalize"):
            values, _ = _l2_column_scale(raw)
        return values

    def extra_repr(self) -> str:
        return (
            f"sigma={self.sigma.item():g}, "
            f"normalize_columns={self.normalize_columns}"
        )

    def tangent_config(self) -> tuple:
        return (self.normalize_columns,)

    @property
    def supports_tangent(self) -> bool:
        return True

    def tangent(self, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
        values, centers, _ = self.tangent_with_precision(
            chart, p, self.sigma.reciprocal().square()
        )
        return values, centers

    def tangent_with_precision(self, chart: Chart, p: Tensor, precision: Tensor):
        """Analytic values, center derivatives and precision derivative."""

        offset, squared, precision = self._geometry(chart, p, precision)
        raw = self._unnormalized_from_squared(squared, precision)
        raw_centers = self._d_raw_d_center(offset, squared, precision)
        raw_widths = self._d_raw_d_precision(squared, precision)
        if not self.normalize_columns:
            return raw, raw_centers, raw_widths
        # Project ∂u in the L2 gauge using 1/||u||, never 1/u. Compact
        # profiles vanish at r=1, so du/u ~ 1/gap diverges while
        # (u/||u||)(du/u) = du/||u|| stays finite. Empty columns stay
        # exactly zero; barely-supported ones use a floor so GH is finite.
        values, scale = _l2_column_scale(raw)
        scale = scale.reshape(1, -1)
        dpsi_dc = raw_centers * scale.unsqueeze(-1)
        dpsi_dprec = raw_widths * scale
        center_mean = (values.unsqueeze(-1) * dpsi_dc).sum(0, keepdim=True)
        centers = dpsi_dc - values.unsqueeze(-1) * center_mean
        precision_mean = (values * dpsi_dprec).sum(0, keepdim=True)
        widths = dpsi_dprec - values * precision_mean
        return values, centers, widths

    def _geometry(
        self, chart: Chart, p: Tensor, precision: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if p.ndim != 2 or p.shape[1] != self.parameter_dim(chart):
            raise ValueError(f"p must have shape [atoms, {self.parameter_dim(chart)}]")
        if precision.ndim not in (0, 1):
            raise ValueError("precision must be scalar or have shape [atoms]")
        if precision.ndim == 1 and precision.shape != (p.shape[0],):
            raise ValueError("precision must be scalar or have shape [atoms]")
        precision = precision.to(device=p.device, dtype=p.dtype)
        offset = chart.center_offsets(p)
        squared = chart.squared_distance(p)
        return offset, squared, precision

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

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        raise NotImplementedError

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        raise NotImplementedError


def _radial_from_squared(squared: Tensor, precision: Tensor) -> Tensor:
    """Return ``q = d/R`` with a floor so ``sqrt`` stays twice differentiable."""

    scaled = squared * precision.reshape(1, -1)
    return (scaled + torch.finfo(scaled.dtype).eps).sqrt()


class WendlandC2(_CompactRadialProfile):
    r"""The Wendland \(C^2\) profile \((1-r)_+^4(4r+1)\), L2-normalized on the chart."""

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        radial = _radial_from_squared(squared, precision)
        gap = (1.0 - radial).clamp_min(0.0)
        return gap.pow(4) * (4.0 * radial + 1.0)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        prec = precision.reshape(1, -1)
        gap = (1.0 - _radial_from_squared(squared, precision)).clamp_min(0.0)
        return (20.0 * gap.pow(3) * prec).unsqueeze(-1) * offset

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        gap = (1.0 - _radial_from_squared(squared, precision)).clamp_min(0.0)
        return -10.0 * squared * gap.pow(3)


class Triangle(_CompactRadialProfile):
    r"""The raw radial triangle profile :math:`(1-r)_+`.

    This is the lowest-order compact radial profile.  Its center derivative
    uses the finite zero subgradient at ``r = 0`` and is discontinuous at the
    support boundary, so it is intended for first-order optimization only.
    """

    normalize_columns = False

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        radial = _radial_from_squared(squared, precision)
        return (1.0 - radial).clamp_min(0.0)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        prec = precision.reshape(1, -1)
        radial = _radial_from_squared(squared, precision)
        active = radial < 1.0
        slope = torch.where(active, prec / radial, torch.zeros_like(radial))
        return slope.unsqueeze(-1) * offset

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        radial = _radial_from_squared(squared, precision)
        active = radial < 1.0
        return torch.where(
            active,
            -0.5 * squared / radial,
            torch.zeros_like(radial),
        )


class Biweight(_CompactRadialProfile):
    r"""The raw biweight profile :math:`(1-r^2)_+^2`.

    The value and first derivative vanish continuously at the support
    boundary.  This is the minimum polynomial order in ``r^2`` suited to a
    first-order optimizer without Triangle's boundary-gradient jump.
    """

    normalize_columns = False

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        return (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0).square()

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        prec = precision.reshape(1, -1)
        inside = (1.0 - squared * prec).clamp_min(0.0)
        return (4.0 * inside * prec).unsqueeze(-1) * offset

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        inside = (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0)
        return -2.0 * squared * inside


class Triweight(_CompactRadialProfile):
    r"""The triweight profile \((1-r^2)_+^3\), L2-normalized by default.

    Set ``normalize_columns=False`` to retain distance and bandwidth
    derivatives when a column is supported by only one site.
    """

    normalize_columns = True

    def _unnormalized_from_squared(self, squared: Tensor, precision: Tensor) -> Tensor:
        return (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0).pow(3)

    def _d_raw_d_center(
        self, offset: Tensor, squared: Tensor, precision: Tensor
    ) -> Tensor:
        prec = precision.reshape(1, -1)
        inside = (1.0 - squared * prec).clamp_min(0.0)
        return (6.0 * inside.square() * prec).unsqueeze(-1) * offset

    def _d_raw_d_precision(self, squared: Tensor, precision: Tensor) -> Tensor:
        inside = (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0)
        return -3.0 * squared * inside.square()


_L2_FLOOR = 1e-6


def _l2_column_scale(values: Tensor) -> tuple[Tensor, Tensor]:
    """L2-normalize live columns; empty columns stay zero.

    The floor keeps ``1/||u||`` inside float32 when a compact atom is
    barely on the chart, without ``detach`` (CUDA graphs cannot capture it).
    Exact zeros remain exact because ``0 / floor = 0``.
    """

    scale = torch.linalg.vector_norm(values, dim=0).clamp_min(_L2_FLOOR).reciprocal()
    return values * scale, scale
