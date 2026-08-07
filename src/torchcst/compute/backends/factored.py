"""The factored backend: apply the measure without ever building W."""

from __future__ import annotations

from torch import Tensor


def apply_rows(x: Tensor, k_in: Tensor, k_out: Tensor, weights: Tensor) -> Tensor:
    """``((x @ k_in) * w) @ k_out.T`` -- the no-W form of the map.

    Autograd retains the ``[rows, K]`` intermediate for backward, which
    is what caps this backend to small K; past the crossover the
    materialized backend is both faster and lighter.
    """
    return ((x @ k_in) * weights) @ k_out.transpose(0, 1)
