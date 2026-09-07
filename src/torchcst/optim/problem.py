"""Pure full-quartic local objective assembled from expanded moments."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from torchcst._derivatives.quadratic import QuadraticFeatureGram

from .moments import ExpandedMoments, MomentContext
from .moments.second import SeparableDiagonalMetric


class QuarticProblem:
    r"""The untruncated second-order represented Adam objective.

    The first-moment term is quadratic in the atom displacement and the metric
    term is quartic. Cross-atom interactions are retained through the complete
    represented displacement passed to the visible metric.
    """

    def __init__(
        self,
        context: MomentContext,
        moments: ExpandedMoments,
        *,
        learning_rate: float,
        evaluation: Literal["auto", "visible", "gram"] = "auto",
    ) -> None:
        if not isinstance(context, MomentContext):
            raise TypeError("context must be a MomentContext")
        if not isinstance(moments, ExpandedMoments):
            raise TypeError("moments must be ExpandedMoments")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if evaluation not in ("auto", "visible", "gram"):
            raise ValueError("evaluation must be 'auto', 'visible', or 'gram'")
        if tuple(moments.second.metric.visible_shape) != context.geometry.visible_shape:
            raise ValueError("second-moment metric does not match visible shape")

        constant = moments.first.corrected.constant.detach().clone()
        linear = moments.first.corrected.linear.detach().clone()
        if constant.shape != context.geometry.point_shape:
            raise ValueError("first moment does not match local point shape")
        if (
            constant.device != context.current_point.device
            or constant.dtype != context.current_point.dtype
        ):
            raise ValueError("first moment must match current point device and dtype")
        linear = 0.5 * (linear + linear.transpose(-1, -2))
        self.context = context
        self.moments = moments
        self.learning_rate = float(learning_rate)
        self._first_constant = constant
        self._first_linear = linear
        self._feature_gram = None
        self.evaluation_backend = "visible"
        atoms, parameters = self.point_shape
        features = atoms * (parameters + parameters * (parameters + 1) // 2)
        supported = (
            context.geometry.supports_quadratic_feature_gram
            and isinstance(moments.second.metric, SeparableDiagonalMetric)
            and constant.dtype in (torch.float32, torch.float64)
        )
        if evaluation == "gram" and not supported:
            raise ValueError(
                "gram evaluation requires factored geometry and a separable metric in float32/64"
            )
        # A100 measurements did not improve the complete FullQuartic solve.
        # Keep GPU use explicit until an end-to-end benefit is established.
        # Bound quadratic storage and avoid setup overhead on tiny visible sites.
        use_gram = evaluation == "gram" or (
            evaluation == "auto"
            and supported
            and constant.device.type == "cpu"
            and features <= 1024
            and atoms
            * sum(context.geometry.visible_shape)
            * (1 + parameters + parameters**2)
            <= 16_000_000
            and math.prod(context.geometry.visible_shape) >= 4 * features
        )
        if use_gram:
            row, column, eps = moments.second.metric.separable_weights()
            self._feature_gram = context.geometry.quadratic_feature_gram(
                point=context.current_point,
                row_weight=row,
                column_weight=column,
                eps=eps,
            )
            if self._feature_gram is None:
                raise RuntimeError(
                    "geometry advertised quadratic Gram support but returned none"
                )
            self.evaluation_backend = "gram"

    @property
    def point_shape(self) -> tuple[int, int]:
        return self.context.geometry.point_shape

    def value(self, displacement: Tensor) -> Tensor:
        """Evaluate the complete quartic objective at ``displacement``."""

        self._validate_local(displacement)
        if self._feature_gram is not None:
            return self._first_value(displacement) + self._feature_gram.value(
                displacement
            ) / (2.0 * self.learning_rate)
        represented = self.context.geometry.displacement(
            displacement,
            point=self.context.current_point,
        )
        first = (self._first_constant * displacement).sum()
        first = first + 0.5 * torch.einsum(
            "kp,kpq,kq->",
            displacement,
            self._first_linear,
            displacement,
        )
        second = self.moments.second.metric.inner(represented, represented)
        return first + second / (2.0 * self.learning_rate)

    def gradient(self, displacement: Tensor) -> Tensor:
        """Evaluate the exact cubic stationarity map of the quartic."""

        self._validate_local(displacement)
        if self._feature_gram is not None:
            return self.value_and_gradient(displacement)[1]
        represented = self.context.geometry.displacement(
            displacement,
            point=self.context.current_point,
        )
        metric_force = self.moments.second.metric.apply(represented)
        first = self._first_constant + torch.einsum(
            "kpq,kq->kp",
            self._first_linear,
            displacement,
        )
        second = self.context.geometry.pullback(
            metric_force,
            point=self.context.current_point,
            displacement=displacement,
        )
        return first + second / self.learning_rate

    def value_and_gradient(self, displacement: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the exact objective and gradient with shared contractions."""

        self._validate_local(displacement)
        if self._feature_gram is not None:
            second_value, second_gradient = self._feature_gram.value_and_gradient(
                displacement
            )
            first_gradient = self._first_constant + torch.einsum(
                "kpq,kq->kp", self._first_linear, displacement
            )
            return (
                self._first_value(displacement)
                + second_value / (2.0 * self.learning_rate),
                first_gradient + second_gradient / (2.0 * self.learning_rate),
            )
        represented = self.context.geometry.displacement(
            displacement,
            point=self.context.current_point,
        )
        metric_force = self.moments.second.metric.apply(represented)
        first_value = (self._first_constant * displacement).sum()
        first_value = first_value + 0.5 * torch.einsum(
            "kp,kpq,kq->",
            displacement,
            self._first_linear,
            displacement,
        )
        value = first_value + (represented * metric_force).sum() / (
            2.0 * self.learning_rate
        )
        first_gradient = self._first_constant + torch.einsum(
            "kpq,kq->kp",
            self._first_linear,
            displacement,
        )
        second_gradient = self.context.geometry.pullback(
            metric_force,
            point=self.context.current_point,
            displacement=displacement,
        )
        return value, first_gradient + second_gradient / self.learning_rate

    def _first_value(self, displacement: Tensor) -> Tensor:
        return (self._first_constant * displacement).sum() + 0.5 * torch.einsum(
            "kp,kpq,kq->", displacement, self._first_linear, displacement
        )

    def hessian(self, displacement: Tensor) -> Tensor:
        """Exact dense Hessian: a weighted frame Gram plus atom-local blocks."""

        self._validate_local(displacement)
        local = self.context.geometry.local_quadratic_derivatives(
            self.context.current_point
        )
        size = displacement.numel()
        if local is None:
            return torch.func.jacfwd(self.gradient)(displacement).reshape(size, size)
        jacobian, second = local
        frame = jacobian + torch.einsum("kmpq,kq->kmp", second, displacement)
        columns = frame.permute(1, 0, 2).reshape(-1, size)
        metric = self.moments.second.metric
        weighted = torch.vmap(metric.apply)(
            columns.T.reshape(size, *self.context.geometry.visible_shape)
        ).reshape(size, -1)
        matrix = columns.T @ weighted.T / self.learning_rate
        represented = self.context.geometry.displacement(
            displacement, point=self.context.current_point
        )
        force = metric.apply(represented).reshape(-1)
        blocks = (
            self._first_linear
            + torch.einsum("m,kmpq->kpq", force, second) / self.learning_rate
        )
        atoms, parameters = self.point_shape
        index = torch.arange(atoms, device=displacement.device)
        matrix.reshape(atoms, parameters, atoms, parameters)[index, :, index, :] += (
            blocks
        )
        return 0.5 * (matrix + matrix.T)

    def restricted_model(self, basis: Tensor) -> _RestrictedQuartic:
        """Exact quartic on d = basis @ y, with small coefficients on the CPU.

        A ball solver must supply orthonormal columns to preserve the radius.
        CPU float64 inner arithmetic reduces launch overhead; coefficients
        retain the accuracy of the original point/derivative dtype.
        """

        point = self.context.current_point
        if basis.ndim != 2 or basis.shape[0] != point.numel() or basis.shape[1] < 1:
            raise ValueError("basis must have shape [local_size, positive_dimension]")
        if basis.dtype != point.dtype or basis.device != point.device:
            raise ValueError("basis must match point dtype and device")
        dimension = basis.shape[1]
        local_basis = basis.reshape(*self.point_shape, dimension)
        derivatives = self.context.geometry.local_quadratic_derivatives(point)
        if derivatives is None:

            def represented(y):
                return self.context.geometry.displacement(
                    (basis @ y).reshape_as(point), point=point
                ).reshape(-1)

            zero = point.new_zeros(dimension)
            first = torch.func.jacfwd(represented)(zero)
            second = torch.func.jacfwd(torch.func.jacfwd(represented))(zero)
        else:
            jacobian, hessian = derivatives
            first = torch.einsum("kmp,kpa->ma", jacobian, local_basis)
            second = torch.einsum(
                "kmpq,kpa,kqb->mab", hessian, local_basis, local_basis
            )
        left, right = torch.triu_indices(dimension, dimension, device=point.device)
        weights = torch.where(left == right, 0.5, 1.0)
        columns = torch.cat((first, second[:, left, right] * weights), dim=1)
        count = columns.shape[1]
        weighted = torch.vmap(self.moments.second.metric.apply)(
            columns.T.reshape(count, *self.context.geometry.visible_shape)
        ).reshape(count, -1)
        gram = columns.T @ weighted.T / self.learning_rate
        a = basis.T @ self._first_constant.reshape(-1)
        b = torch.einsum(
            "kpa,kpq,kqb->ab", local_basis, self._first_linear, local_basis
        )
        return _RestrictedQuartic(
            a.detach().cpu().double(),
            b.detach().cpu().double(),
            gram.detach().cpu().double(),
        )

    def _validate_local(self, value: Tensor) -> None:
        if value.shape != self.point_shape:
            raise ValueError(f"displacement must have shape {list(self.point_shape)}")
        point = self.context.current_point
        if value.device != point.device or value.dtype != point.dtype:
            raise ValueError("displacement must match current point device and dtype")


class _RestrictedQuartic:
    def __init__(self, linear: Tensor, quadratic: Tensor, gram: Tensor) -> None:
        self.linear = linear
        self.quadratic = 0.5 * (quadratic + quadratic.T)
        self.gram = QuadraticFeatureGram(0.5 * (gram + gram.T), (1, linear.numel()))

    def value_and_gradient(self, d: Tensor) -> tuple[Tensor, Tensor]:
        norm, gradient = self.gram.value_and_gradient(d)
        flat = d.reshape(-1)
        first_gradient = self.linear + self.quadratic @ flat
        value = self.linear @ flat + 0.5 * flat @ self.quadratic @ flat + 0.5 * norm
        return value, first_gradient.reshape_as(d) + 0.5 * gradient

    def hessian(self, d: Tensor) -> Tensor:
        return self.quadratic + 0.5 * self.gram.hessian(d)
