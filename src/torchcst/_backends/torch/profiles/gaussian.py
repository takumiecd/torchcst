"""PyTorch execution for declared gaussian contracts."""

from __future__ import annotations

import torch
from torch import Tensor

from torchcst._backends.torch.profiles import execution as _profile
from torchcst.geometry import Chart
from torchcst.kernels.spec import AtomInit


def initialize(state, chart: Chart, atoms: int, *, mode: AtomInit) -> Tensor:
    return chart.initialize_centers(atoms, mode=mode)


def evaluate(state, chart: Chart, p: Tensor) -> Tensor:
    precision = state.sigma.reciprocal().square()
    return _profile.evaluate_with_precision(state, chart, p, precision)


def evaluate_with_precision(
    state,
    chart: Chart,
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
    squared_distance = chart.squared_distance(p)
    log_squared_values = -squared_distance * precision
    if state.binding.normalization.kind == "none":
        return torch.exp(0.5 * log_squared_values)
    return torch.exp(0.5 * torch.log_softmax(log_squared_values, dim=0))


def evaluate_with_precision_slice(
    state,
    chart: Chart,
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
    squared = chart.squared_distance(p, selection)
    return torch.exp(-0.5 * squared * precision.to(device=p.device, dtype=p.dtype))


def tangent(state, chart: Chart, p: Tensor) -> tuple[Tensor, Tensor]:
    values, centers, _ = _profile.tangent_with_precision(
        state, chart, p, state.sigma.reciprocal().square()
    )
    return values, centers


def tangent_with_precision(state, chart: Chart, p: Tensor, precision: Tensor):
    """Analytic values, center derivatives and precision derivative.

    Includes the complete L2 normalization, with no amplitude division.
    """
    precision = precision.to(p)
    values = _profile.evaluate_with_precision(state, chart, p, precision)
    offset = chart.center_offsets(p)
    squared = chart.squared_distance(p)
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


def project_gradient(state, chart: Chart, p: Tensor, gradient: Tensor) -> Tensor:
    return chart.geometry.project_tangent(p, gradient)


def apply_parameter_update(
    state,
    chart: Chart,
    p: Tensor,
    displacement: Tensor,
) -> Tensor:
    return chart.geometry.retract(p, displacement)


def transport_state(
    kernel_state,
    chart: Chart,
    old: Tensor,
    new: Tensor,
    state: Tensor,
) -> Tensor:
    return chart.geometry.transport(old, new, state)
