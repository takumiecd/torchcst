"""Structural contract used to discover CST sites."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn

from .atoms import AtomDerivatives


@runtime_checkable
class CSTSite(Protocol):
    """A module exposing one atom table and its represented cotangent."""

    def cst_parameters(self) -> tuple[nn.Parameter, ...]: ...

    def cst_derivatives(self) -> AtomDerivatives: ...

    def enable_represented_gradient_capture(self, *, clear: bool = True) -> None: ...

    def disable_represented_gradient_capture(self) -> None: ...

    def clear_represented_gradient(self) -> None: ...

    def represented_gradient(self) -> Tensor: ...
