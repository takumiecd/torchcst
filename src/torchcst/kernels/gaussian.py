"""Isotropic Gaussian kernel."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import Kernel


class Gaussian(Kernel):
    r"""The isotropic kernel ``exp(-||u-v||² / (2 sigma²))``."""

    def __init__(self, sigma: float | Tensor, *, trainable: bool = False) -> None:
        super().__init__()
        value = torch.as_tensor(sigma)
        if value.numel() != 1:
            raise ValueError("sigma must be a scalar")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        value = value.detach().clone().reshape(())
        if not torch.isfinite(value) or value <= 0:
            raise ValueError("sigma must be finite and positive")

        if trainable:
            self._log_sigma = nn.Parameter(value.log())
            self.register_buffer("_fixed_sigma", None)
        else:
            self.register_parameter("_log_sigma", None)
            self.register_buffer("_fixed_sigma", value)

    @property
    def sigma(self) -> Tensor:
        """The positive bandwidth tensor."""

        if self._log_sigma is not None:
            return self._log_sigma.exp()
        return self._fixed_sigma

    @property
    def trainable(self) -> bool:
        return self._log_sigma is not None

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim < 2 or right.ndim < 2:
            raise ValueError("kernel inputs must have shape [..., points, dimensions]")
        if left.shape[-1] != right.shape[-1]:
            raise ValueError("kernel inputs must have the same coordinate dimension")
        if not self.supports_dimension(left.shape[-1]):
            raise ValueError("unsupported coordinate dimension")

        squared_distance = (
            left.unsqueeze(-2) - right.unsqueeze(-3)
        ).square().sum(dim=-1)
        return torch.exp(-squared_distance / (2 * self.sigma.square()))

    def extra_repr(self) -> str:
        sigma = math.exp(self._log_sigma.item()) if self.trainable else self.sigma.item()
        return f"sigma={sigma:g}, trainable={self.trainable}"
