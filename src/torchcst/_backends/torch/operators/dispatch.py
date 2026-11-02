"""PyTorch reference implementations for Linear execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from torchcst._backends.torch.kernels import execution as _kernel
from torchcst.charts import ChartState
from torchcst.kernels.state import KernelState

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear


def validate_materialized(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    pass


def validate_factored(charts: tuple[ChartState, ...], kernel: KernelState) -> None:
    if not _kernel.supports_factorization(kernel):
        raise ValueError("the selected kernel does not support factorized execution")
    _kernel.validate_layout(kernel, charts)


def materialized(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="materialized")


def factored(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="factored")
