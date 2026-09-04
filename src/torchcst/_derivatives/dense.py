"""Dense autograd oracle for atom-structured derivatives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import jacrev

from .atoms import AtomDerivatives


class DenseDerivativeOracle(AtomDerivatives):
    """Materialize full derivative tensors for small correctness tests only."""

    def jacobian(self, *, parameter_point: Tensor | None = None) -> Tensor:
        """Return ``J`` with shape ``[*visible_shape, K, Q]``."""

        parameter_point = self._point_or_current(parameter_point)
        return jacrev(self.represented)(parameter_point)

    def hessian(self, *, parameter_point: Tensor | None = None) -> Tensor:
        """Return ``H`` with shape ``[*visible_shape, K, Q, K, Q]``."""

        parameter_point = self._point_or_current(parameter_point)
        return jacrev(jacrev(self.represented))(parameter_point)

    def full_contracted_hessian(
        self, cotangent: Tensor, *, parameter_point: Tensor | None = None
    ) -> Tensor:
        """Return the dense contracted Hessian with shape ``[K, Q, K, Q]``."""

        parameter_point = self._point_or_current(parameter_point)
        self._validate_cotangent(cotangent, parameter_point)
        visible_dims = tuple(range(cotangent.ndim))
        return torch.tensordot(
            cotangent,
            self.hessian(parameter_point=parameter_point),
            dims=(visible_dims, visible_dims),
        )
