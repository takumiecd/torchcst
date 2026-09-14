"""Small functional Hessian helpers used by derivative observers."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import jvp, vmap


def hessian_from_gradient(
    gradient,
    point: Tensor,
    *,
    directions: Tensor | None = None,
) -> Tensor:
    """Materialize one local Hessian by JVPs of a gradient function.

    ``gradient`` maps one parameter vector ``[P]`` to a gradient with shape
    ``[P]``. The returned tensor has shape ``[P, P]``. Its last dimension is
    the differentiated direction. Only the small local Hessian is built;
    output-space and global Hessians are never materialized here.
    """

    if point.ndim != 1:
        raise ValueError("point must be a one-dimensional parameter vector")
    if directions is None:
        directions = torch.eye(
            point.shape[0], device=point.device, dtype=point.dtype
        )
    if directions.shape != (point.shape[0], point.shape[0]):
        raise ValueError("directions must have shape [P, P]")
    if directions.device != point.device or directions.dtype != point.dtype:
        raise ValueError("directions must match point device and dtype")
    columns = vmap(
        lambda direction: jvp(gradient, (point,), (direction,))[1]
    )(directions)
    return columns.transpose(-1, -2)
