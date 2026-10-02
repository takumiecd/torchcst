"""PyTorch execution for declared gaussian contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.profiles import execution as _profile
from torchcst.charts import ChartState
from torchcst.kernels.spec import AtomInit


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
    """Evaluate normalized columns with scalar or per-atom precision."""

    if p.ndim != 2 or p.shape[1] != _profile.parameter_dim(state, chart):
        raise ValueError(
            f"p must have shape [atoms, {_profile.parameter_dim(state, chart)}]"
        )
    if precision.ndim not in (0, 1):
        raise ValueError("precision must be scalar or have shape [atoms]")
    if precision.ndim == 1 and precision.shape != (p.shape[0],):
        raise ValueError("precision must be scalar or have shape [atoms]")
    precision = precision.to(device=p.device, dtype=p.dtype)
    squared_distance = _charts.squared_distance(chart, p)
    log_squared_values = -squared_distance * precision
    if state.binding.normalization.kind == "none":
        return torch.exp(0.5 * log_squared_values)
    return torch.exp(0.5 * torch.log_softmax(log_squared_values, dim=0))


def evaluate_with_precision_slice(
    state,
    chart: ChartState,
    p: Tensor,
    precision: Tensor,
    selection: slice | Tensor,
) -> Tensor:
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
    return torch.exp(-0.5 * squared * precision.to(device=p.device, dtype=p.dtype))


def tangent(state, chart: ChartState, p: Tensor) -> tuple[Tensor, Tensor]:
    values, centers, _ = _profile.tangent_with_precision(
        state, chart, p, state.sigma.reciprocal().square()
    )
    return values, centers


def tangent_with_precision(state, chart: ChartState, p: Tensor, precision: Tensor):
    """Analytic values, center derivatives and precision derivative.

    Includes the complete L2 normalization, with no amplitude division.
    """
    precision = precision.to(p)
    values = _profile.evaluate_with_precision(state, chart, p, precision)
    offset = _charts.center_offsets(chart, p)
    squared = _charts.squared_distance(chart, p)
    if state.binding.normalization.kind == "none":
        centers = values[..., None] * precision.reshape(1, -1, 1) * offset
        widths = -0.5 * values * squared
        return values, centers, widths
    probability = values.square()
    center_log = 2 * precision.reshape(1, -1, 1) * offset
    center_mean = (probability[..., None] * center_log).sum(0, keepdim=True)
    centers = 0.5 * values[..., None] * (center_log - center_mean)
    precision_mean = (probability * squared).sum(0, keepdim=True)
    widths = 0.5 * values * (precision_mean - squared)
    return values, centers, widths


def project_gradient(state, chart: ChartState, p: Tensor, gradient: Tensor) -> Tensor:
    return _geometry.project_tangent(chart.geometry, p, gradient)


def apply_parameter_update(
    state,
    chart: ChartState,
    p: Tensor,
    displacement: Tensor,
) -> Tensor:
    return _geometry.retract(chart.geometry, p, displacement)


def transport_state(
    kernel_state,
    chart: ChartState,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    return _geometry.transport(chart.geometry, old, new, state)
