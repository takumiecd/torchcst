"""Opaque, fixed-shape atom parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .grad import AtomGrad


class Atoms(nn.Module):
    """Own one opaque, fixed-shape parameter row per atom.

    ``Atoms`` deliberately assigns no meaning to columns of ``p``. The selected
    kernel is the only component allowed to interpret them.
    """

    def __init__(self, p: Tensor) -> None:
        super().__init__()
        if not isinstance(p, Tensor):
            raise TypeError("p must be a torch.Tensor")
        if p.ndim != 2:
            raise ValueError("p must have shape [atoms, parameter_dim]")
        if p.shape[0] < 1:
            raise ValueError("Atoms must contain at least one atom")
        if p.shape[1] < 1:
            raise ValueError("p must contain at least one coordinate per atom")
        if not p.is_floating_point():
            raise TypeError("p must have a floating-point dtype")
        if not torch.isfinite(p).all():
            raise ValueError("p must be finite")

        self.p = nn.Parameter(p.detach().clone())
        self.__dict__["_grad"] = None

    @property
    def count(self) -> int:
        """The fixed number of atoms."""

        return self.p.shape[0]

    @property
    def parameter_dim(self) -> int:
        """The opaque kernel-coordinate width ``P``."""

        return self.p.shape[1]

    @property
    def grad(self) -> AtomGrad | None:
        """The optimizer-provided, transient atom-gradient program."""

        return self.__dict__["_grad"]

    def set_grad(self, grad: AtomGrad | None) -> None:
        """Attach an optimizer gradient program without registering model state."""

        if grad is not None and not isinstance(grad, AtomGrad):
            raise TypeError("grad must be an AtomGrad or None")
        current = self.grad
        if current is grad:
            return
        if current is not None:
            current._detach(self)
        if grad is not None:
            grad._attach(self)
        self.__dict__["_grad"] = grad

    def extra_repr(self) -> str:
        return f"count={self.count}, parameter_dim={self.parameter_dim}"
