"""Opaque, fixed-shape atom parameters."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Atoms(nn.Module):
    """Own amplitudes and opaque kernel coordinates for a fixed atom set.

    ``Atoms`` deliberately assigns no meaning to columns of ``p``. The selected
    kernel is the only component allowed to interpret them.
    """

    def __init__(self, weight: Tensor, p: Tensor) -> None:
        super().__init__()
        if not isinstance(weight, Tensor) or not isinstance(p, Tensor):
            raise TypeError("weight and p must be torch.Tensor instances")
        if weight.ndim != 1:
            raise ValueError("weight must have shape [atoms]")
        if p.ndim != 2:
            raise ValueError("p must have shape [atoms, parameter_dim]")
        if weight.shape[0] < 1:
            raise ValueError("Atoms must contain at least one atom")
        if p.shape[0] != weight.shape[0]:
            raise ValueError("weight and p must contain the same number of atoms")
        if p.shape[1] < 1:
            raise ValueError("p must contain at least one coordinate per atom")
        if not weight.is_floating_point() or not p.is_floating_point():
            raise TypeError("weight and p must have floating-point dtypes")
        if weight.device != p.device or weight.dtype != p.dtype:
            raise ValueError("weight and p must share one device and dtype")
        if not torch.isfinite(weight).all() or not torch.isfinite(p).all():
            raise ValueError("weight and p must be finite")

        self.weight = nn.Parameter(weight.detach().clone())
        self.p = nn.Parameter(p.detach().clone())

    @property
    def count(self) -> int:
        """The fixed number of atoms."""

        return self.weight.shape[0]

    @property
    def parameter_dim(self) -> int:
        """The opaque kernel-coordinate width ``P``."""

        return self.p.shape[1]

    @property
    def local_parameter_dim(self) -> int:
        """Per-atom optimizer width, including the amplitude."""

        return self.parameter_dim + 1

    def local_parameters(self) -> Tensor:
        """Return the structured ``[K, P + 1]`` local parameter table."""

        return torch.cat((self.weight.unsqueeze(-1), self.p), dim=-1)

    def extra_repr(self) -> str:
        return f"count={self.count}, parameter_dim={self.parameter_dim}"
