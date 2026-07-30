"""Shared data-moment assembly. Private to torchcst; imports only torch."""

from __future__ import annotations

import torch
from torch import Tensor


def data_second_moment(
    data: Tensor,
    *,
    n_in: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Validate an ``[n, n_in]`` batch and return ``Sigma_x = X^T X / n``.

    The one D-metric assembly shared by :class:`~torchcst.representation.gram.
    GramService` and the init-placement functions, so the convention cannot
    drift between them.
    """
    if not isinstance(data, Tensor):
        raise TypeError("data must be a Tensor or None")
    if data.ndim != 2 or data.shape[1] != n_in:
        raise ValueError("data must have shape [n, n_in] matching V's rows")
    if not data.is_floating_point():
        raise TypeError("data must have a floating dtype")
    if data.shape[0] < 1:
        raise ValueError("data must have at least one row")
    data = data.detach().to(device=device, dtype=dtype)
    return (data.transpose(0, 1) @ data) / data.shape[0]
