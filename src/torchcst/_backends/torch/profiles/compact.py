"""PyTorch execution for declared compact contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry_execution
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry.state import ChartState
from torchcst.kernels.spec import AtomInit
from torchcst.profiling import cst_span


def initialize(state, chart: ChartState, atoms: int, *, mode: AtomInit) -> Tensor:
    return _charts.initialize_centers(chart, atoms, mode=mode)


def evaluate(state, chart: ChartState, p: Tensor) -> Tensor:
    precision = state.sigma.reciprocal().square()
    return _profile.evaluate_with_precision(state, chart, p, precision)


def evaluate_with_precision(
    state,
    chart: ChartState,
    p: Tensor,
    precision: Tensor,
) -> Tensor:
    with cst_span("cst.profile.geometry"):
        _, squared, precision = _geometry(
            state, chart, p, precision, need_offsets=False
        )
    with cst_span("cst.profile.radial"):
        raw = _unnormalized_from_squared(state, squared, precision)
    if state.binding.normalization.kind == "none":
        return raw
    with cst_span("cst.profile.normalize"):
        values, _ = _l2_column_scale(raw)
    return values


def evaluate_with_precision_slice(
    state,
    chart: ChartState,
    p: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
    """Evaluate raw compact values for a bounded set of operator sites."""

    if state.binding.normalization.kind != "none":
        raise ValueError("sliced evaluation requires normalize_columns=False")
    if p.ndim != 2 or p.shape[1] != _profile.parameter_dim(state, chart):
        raise ValueError(
            f"p must have shape [atoms, {_profile.parameter_dim(state, chart)}]"
        )
    if precision.ndim not in (0, 1) or (
        precision.ndim == 1 and precision.shape != (p.shape[0],)
    ):
        raise ValueError("precision must be scalar or have shape [atoms]")
    squared = _charts.squared_distance(chart, p, selection)
    return _unnormalized_from_squared(
        state, squared, precision.to(device=p.device, dtype=p.dtype)
    )


def tangent(state, chart: ChartState, p: Tensor) -> tuple[Tensor, Tensor]:
    values, centers, _ = _profile.tangent_with_precision(
        state, chart, p, state.sigma.reciprocal().square()
    )
    return values, centers


def tangent_with_precision(state, chart: ChartState, p: Tensor, precision: Tensor):
    """Analytic values, center derivatives and precision derivative."""

    offset, squared, precision = _geometry(state, chart, p, precision)
    raw = _unnormalized_from_squared(state, squared, precision)
    raw_centers = _d_raw_d_center(state, offset, squared, precision)
    raw_widths = _d_raw_d_precision(state, squared, precision)
    if state.binding.normalization.kind == "none":
        return raw, raw_centers, raw_widths
    # Project ∂u in the L2 gauge using 1/||u||, never 1/u. Compact
    # profiles vanish at r=1, so du/u ~ 1/gap diverges while
    # (u/||u||)(du/u) = du/||u|| stays finite. Empty columns stay
    # exactly zero; barely-supported ones use a floor so GH is finite.
    values, scale = _l2_column_scale(raw)
    scale = scale.reshape(1, -1)
    dpsi_dc = raw_centers * scale.unsqueeze(-1)
    dpsi_dprec = raw_widths * scale
    # clamp_min has zero denominator derivative below the norm floor,
    # and its ordinary derivative at equality. Keep that forward contract.
    live_norm = (torch.linalg.vector_norm(raw, dim=0) >= _L2_FLOOR).reshape(1, -1)
    center_mean = (values.unsqueeze(-1) * dpsi_dc).sum(0, keepdim=True)
    center_mean = center_mean * live_norm.unsqueeze(-1)
    centers = dpsi_dc - values.unsqueeze(-1) * center_mean
    precision_mean = (values * dpsi_dprec).sum(0, keepdim=True)
    precision_mean = precision_mean * live_norm
    widths = dpsi_dprec - values * precision_mean
    return values, centers, widths


def _geometry(
    state, chart: ChartState, p: Tensor, precision: Tensor, *, need_offsets: bool = True
) -> tuple[Tensor | None, Tensor, Tensor]:
    if p.ndim != 2 or p.shape[1] != _profile.parameter_dim(state, chart):
        raise ValueError(
            f"p must have shape [atoms, {_profile.parameter_dim(state, chart)}]"
        )
    if precision.ndim not in (0, 1):
        raise ValueError("precision must be scalar or have shape [atoms]")
    if precision.ndim == 1 and precision.shape != (p.shape[0],):
        raise ValueError("precision must be scalar or have shape [atoms]")
    precision = precision.to(device=p.device, dtype=p.dtype)
    offset = None
    if need_offsets:
        with cst_span("cst.profile.center_offsets"):
            offset = _charts.center_offsets(chart, p)
    with cst_span("cst.profile.squared_distance"):
        squared = _charts.squared_distance(chart, p)
    return offset, squared, precision


def project_gradient(state, chart: ChartState, p: Tensor, gradient: Tensor) -> Tensor:
    return _geometry_execution.project_tangent(chart.geometry, p, gradient)


def apply_parameter_update(
    state,
    chart: ChartState,
    p: Tensor,
    displacement: Tensor,
) -> Tensor:
    return _geometry_execution.retract(chart.geometry, p, displacement)


def transport_state(
    kernel_state,
    chart: ChartState,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    return _geometry_execution.transport(chart.geometry, old, new, state)


def _unnormalized_from_squared(state, squared: Tensor, precision: Tensor) -> Tensor:
    return _profile.shape_function(
        state, "unnormalized_from_squared", squared, precision
    )


def _d_raw_d_center(
    state, offset: Tensor, squared: Tensor, precision: Tensor
) -> Tensor:
    return _profile.shape_function(state, "d_raw_d_center", offset, squared, precision)


def _d_raw_d_precision(state, squared: Tensor, precision: Tensor) -> Tensor:
    return _profile.shape_function(state, "d_raw_d_precision", squared, precision)


def wendlandc2_unnormalized_from_squared(
    state, squared: Tensor, precision: Tensor
) -> Tensor:
    radial = _radial_from_squared(squared, precision)
    gap = (1.0 - radial).clamp_min(0.0)
    return gap.pow(4) * (4.0 * radial + 1.0)


def wendlandc2_d_raw_d_center(
    state, offset: Tensor, squared: Tensor, precision: Tensor
) -> Tensor:
    prec = precision.reshape(1, -1)
    gap = (1.0 - _radial_from_squared(squared, precision)).clamp_min(0.0)
    return (20.0 * gap.pow(3) * prec).unsqueeze(-1) * offset


def wendlandc2_d_raw_d_precision(state, squared: Tensor, precision: Tensor) -> Tensor:
    gap = (1.0 - _radial_from_squared(squared, precision)).clamp_min(0.0)
    return -10.0 * squared * gap.pow(3)


def triangle_unnormalized_from_squared(
    state, squared: Tensor, precision: Tensor
) -> Tensor:
    radial = _radial_from_squared(squared, precision)
    return (1.0 - radial).clamp_min(0.0)


def triangle_d_raw_d_center(
    state, offset: Tensor, squared: Tensor, precision: Tensor
) -> Tensor:
    prec = precision.reshape(1, -1)
    radial = _radial_from_squared(squared, precision)
    active = radial < 1.0
    slope = torch.where(active, prec / radial, torch.zeros_like(radial))
    return slope.unsqueeze(-1) * offset


def triangle_d_raw_d_precision(state, squared: Tensor, precision: Tensor) -> Tensor:
    radial = _radial_from_squared(squared, precision)
    active = radial < 1.0
    return torch.where(
        active,
        -0.5 * squared / radial,
        torch.zeros_like(radial),
    )


def biweight_unnormalized_from_squared(
    state, squared: Tensor, precision: Tensor
) -> Tensor:
    return (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0).square()


def biweight_d_raw_d_center(
    state, offset: Tensor, squared: Tensor, precision: Tensor
) -> Tensor:
    prec = precision.reshape(1, -1)
    inside = (1.0 - squared * prec).clamp_min(0.0)
    return (4.0 * inside * prec).unsqueeze(-1) * offset


def biweight_d_raw_d_precision(state, squared: Tensor, precision: Tensor) -> Tensor:
    inside = (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0)
    return -2.0 * squared * inside


def triweight_unnormalized_from_squared(
    state, squared: Tensor, precision: Tensor
) -> Tensor:
    return (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0).pow(3)


def triweight_d_raw_d_center(
    state, offset: Tensor, squared: Tensor, precision: Tensor
) -> Tensor:
    prec = precision.reshape(1, -1)
    inside = (1.0 - squared * prec).clamp_min(0.0)
    return (6.0 * inside.square() * prec).unsqueeze(-1) * offset


def triweight_d_raw_d_precision(state, squared: Tensor, precision: Tensor) -> Tensor:
    inside = (1.0 - squared * precision.reshape(1, -1)).clamp_min(0.0)
    return -3.0 * squared * inside.square()


def _radial_from_squared(squared: Tensor, precision: Tensor) -> Tensor:
    """Return ``q = d/R`` with a floor so ``sqrt`` stays twice differentiable."""

    scaled = squared * precision.reshape(1, -1)
    return (scaled + torch.finfo(scaled.dtype).eps).sqrt()


_L2_FLOOR = 1e-6


def _l2_column_scale(values: Tensor) -> tuple[Tensor, Tensor]:
    """L2-normalize live columns; empty columns stay zero.

    The floor keeps ``1/||u||`` inside float32 when a compact atom is
    barely on the chart, without ``detach`` (CUDA graphs cannot capture it).
    Exact zeros remain exact because ``0 / floor = 0``.
    """

    scale = torch.linalg.vector_norm(values, dim=0).clamp_min(_L2_FLOOR).reciprocal()
    return values * scale, scale
