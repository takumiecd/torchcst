"""PyTorch reference implementations for Linear execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from torchcst.geometry import Chart
from torchcst.kernels import Kernel
from torchcst.profiling import cst_span

from .._strip_torus import tiled_linear
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
    return site.operator.apply(inputs, p, algorithm="materialized")


def factored(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="factored")


def tiled(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    with cst_span("cst.linear.route_atoms"):
        from .._support_layout import support_layout
        from ._preparation import execution_plan

        plan = execution_plan(site)
        encoded, _, precision = site.kernel.tile_parameters(site.chart, p)
        decoded = site.chart.geometry.decode_centers(encoded)
        layout = support_layout(plan, decoded, precision, site.chart.tile_shape[0])
    with cst_span("cst.linear.tiled_matmul"):
        return tiled_linear(
            site.chart, site.kernel, inputs, p, layout, support_layout=True
        )
