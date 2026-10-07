"""PyTorch reference implementations for Linear execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.operators.strip_torus.tiled import tiled_linear
from torchcst._backends.torch.operators.strip_torus.tiled import (
    validate_tiled as validate_strip_torus,
)
from torchcst._backends.torch.parameterizations import direct_amp_width as _coordinates
from torchcst.charts import ChartState
from torchcst.kernels.state import KernelState
from torchcst.profiling import cst_span

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear


def validate_materialized(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    pass


def validate_factored(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    if not _kernel.supports_factorization(kernel):
        raise ValueError("the selected kernel does not support factorized execution")
    _kernel.validate_layout(kernel, charts)


def validate_tiled(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    if len(charts) != 1:
        raise ValueError("tiled backend requires a single operator chart")
    validate_strip_torus(charts[0], kernel)


def materialized(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="materialized")


def factored(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="factored")


def tiled(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    with cst_span("cst.linear.route_atoms"):
        from torchcst._backends.torch.operators.strip_torus.preparation import (
            execution_plan,
        )
        from torchcst._backends.torch.operators.strip_torus.support import (
            support_layout,
        )

        plan = execution_plan(site)
        encoded, _, precision = _coordinates.tile_parameters(site.kernel, site.chart, p)
        decoded = _geometry.decode_centers(site.chart.geometry, encoded)
        layout = support_layout(plan, decoded, precision, site.chart.tile_shape[0])
    with cst_span("cst.linear.tiled_matmul"):
        return tiled_linear(
            site.chart, site.kernel, inputs, p, layout, support_layout=True
        )
