"""Dense autograd oracle for local CST derivatives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import jacrev

from torchcst.nn import CSTLinear

from .linear import LinearDerivatives


class DenseDerivativeOracle(LinearDerivatives):
    """Materialize full derivative tensors for small correctness tests only."""

    def __init__(self, site: CSTLinear) -> None:
        super().__init__(site)

    def jacobian(self, *, point: Tensor | None = None) -> Tensor:
        """Return ``J`` with shape ``[out, in, parameters]``."""

        point = self._point_or_current(point)
        return jacrev(self.represented)(point)

    def hessian(self, *, point: Tensor | None = None) -> Tensor:
        """Return ``H`` with shape ``[out, in, parameters, parameters]``."""

        point = self._point_or_current(point)
        return jacrev(jacrev(self.represented))(point)

    def jvp(self, direction: Tensor, *, point: Tensor | None = None) -> Tensor:
        point = self._point_or_current(point)
        self.layout.validate(direction, name="direction")
        return torch.einsum("oip,p->oi", self.jacobian(point=point), direction)

    def second(
        self,
        left: Tensor,
        right: Tensor,
        *,
        point: Tensor | None = None,
    ) -> Tensor:
        point = self._point_or_current(point)
        self.layout.validate(left, name="left direction")
        self.layout.validate(right, name="right direction")
        return torch.einsum(
            "oipq,p,q->oi", self.hessian(point=point), left, right
        )

    def pullback(
        self,
        cotangent: Tensor,
        *,
        at: Tensor | None = None,
        point: Tensor | None = None,
    ) -> Tensor:
        point = self._point_or_current(point)
        expected_shape = (self.site.out_features, self.site.in_features)
        if cotangent.shape != expected_shape:
            raise ValueError(f"cotangent must have shape {list(expected_shape)}")
        if cotangent.device != point.device or cotangent.dtype != point.dtype:
            raise ValueError("cotangent must match the CST site device and dtype")
        if at is None:
            at = torch.zeros_like(point)
        else:
            self.layout.validate(at, name="linearization direction")

        jacobian_pullback = torch.einsum(
            "oip,oi->p", self.jacobian(point=point), cotangent
        )
        hessian_pullback = torch.einsum(
            "oipq,p,oi->q", self.hessian(point=point), at, cotangent
        )
        return jacobian_pullback + hessian_pullback
