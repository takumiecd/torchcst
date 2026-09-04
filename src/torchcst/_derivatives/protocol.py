"""Structural contract used to discover CST sites."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import nn

from torchcst.atoms import Atoms

from .atoms import AtomDerivatives
from .frame import FrameGeometry


@runtime_checkable
class CSTSite(Protocol):
    """A module exposing one atom table and its derivative operators."""

    atoms: Atoms

    def cst_parameters(self) -> tuple[nn.Parameter, ...]: ...

    def cst_derivatives(self) -> AtomDerivatives: ...

    def cst_frame_geometry(self) -> FrameGeometry: ...
