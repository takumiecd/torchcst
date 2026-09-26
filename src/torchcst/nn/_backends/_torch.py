"""PyTorch reference implementations for Linear execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from torchcst.geometry import Chart
from torchcst.kernels import Kernel
from torchcst.profiling import cst_span

from .._layout import initial_layout
from .._strip_torus import route_atoms, tiled_linear
from .._strip_torus import validate_tiled as validate_strip_torus

if TYPE_CHECKING:
    from ..linear import CSTLinear


def validate_materialized(charts: tuple[Chart, ...], kernel: Kernel) -> None:
    pass


def validate_factored(charts: tuple[Chart, ...], kernel: Kernel) -> None:
    if len(charts) != 2 or not kernel.supports_factorization:
        raise ValueError("the selected kernel does not support factorized execution")


def validate_tiled(charts: tuple[Chart, ...], kernel: Kernel) -> None:
    if len(charts) != 1:
        raise ValueError("tiled backend requires a single operator chart")
    validate_strip_torus(charts[0], kernel)


def materialized(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    with cst_span("cst.linear.materialize"):
        charts = site.cst_charts()
        weight = (
            site.kernel.weight(charts[0], p)
            if len(charts) == 1
            else site._materialize_atoms(p).sum(dim=0)
        )
    with cst_span("cst.linear.dense_matmul"):
        return F.linear(inputs, weight)


def factored(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    with cst_span("cst.linear.factors"):
        phi_input, phi_output = site.kernel.factors(*site.cst_charts(), p)
    with cst_span("cst.linear.input_matmul"):
        atom_values = inputs @ phi_input
    with cst_span("cst.linear.output_matmul"):
        return atom_values @ phi_output.transpose(-2, -1)


def tiled(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    with cst_span("cst.linear.route_atoms"):
        owners = route_atoms(site.chart, site.kernel, p)
        layout = initial_layout(owners, site.chart.tile_count)
    with cst_span("cst.linear.tiled_matmul"):
        return tiled_linear(site.chart, site.kernel, inputs, p, layout)
