"""Continuous kernels for the general CST composition path.

Delta and dot products are intentionally absent here: they are the implicit
specialized kernels implemented by :class:`torchcst.compute.EntryLinear` and
:class:`torchcst.compute.RankOneLinear`, respectively.  Keeping those paths
specialized avoids materializing general kernel matrices for discrete entry
and factorized rank-one families.
"""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn


class GaussianKernel(nn.Module):
    """Global-bandwidth isotropic Gaussian kernel.

    ``kappa(query, centers) = exp(-||query-center||^2 / (2 sigma^2))``.
    The scalar ``sigma`` is an ``nn.Parameter`` when ``learnable=True`` and a
    buffer otherwise.  Per-atom bandwidths are outside step 9; a future v0.3
    implementation may revive the old ``SynapseStore.add_extra`` design for
    that purpose.
    """

    def __init__(self, sigma: float | Tensor, learnable: bool = True) -> None:
        super().__init__()
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a bool")
        if isinstance(sigma, bool):
            raise TypeError("sigma must be a real scalar")
        value = torch.as_tensor(sigma)
        if value.numel() != 1 or value.is_complex():
            raise ValueError("sigma must be a real scalar")
        if not value.is_floating_point():
            value = value.to(dtype=torch.get_default_dtype())
        value = value.detach().reshape(()).clone()
        reading = float(value)
        if not isfinite(reading) or reading <= 0:
            raise ValueError("sigma must be finite and positive")
        if learnable:
            self.sigma = nn.Parameter(value)
        else:
            self.register_buffer("sigma", value)

    @property
    def learnable(self) -> bool:
        return isinstance(self.sigma, nn.Parameter)

    def forward(self, query: Tensor, centers: Tensor) -> Tensor:
        if not isinstance(query, Tensor) or not isinstance(centers, Tensor):
            raise TypeError("query and centers must be Tensors")
        if query.ndim != 2 or centers.ndim != 2:
            raise ValueError("query and centers must be rank-2 Tensors")
        if query.shape[1] != centers.shape[1]:
            raise ValueError("query and centers must share their coordinate dimension")
        if not query.is_floating_point() or not centers.is_floating_point():
            raise TypeError("Gaussian coordinates must have floating dtypes")
        sigma = self.sigma.to(device=query.device, dtype=query.dtype)
        if not bool(torch.isfinite(sigma)) or bool(sigma <= 0):
            raise ValueError("sigma must remain finite and positive")
        centers = centers.to(device=query.device, dtype=query.dtype)
        squared_distance = (query[:, None, :] - centers[None, :, :]).square().sum(-1)
        return torch.exp(-squared_distance / (2.0 * sigma.square()))
