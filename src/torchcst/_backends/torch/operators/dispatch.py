"""PyTorch reference implementations for Linear execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from torchcst.nn.linear import CSTLinear


def materialized(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="materialized")


def factored(site: CSTLinear, inputs: Tensor, p: Tensor) -> Tensor:
    return site.operator.apply(inputs, p, algorithm="factored")
