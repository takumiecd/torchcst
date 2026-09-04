"""Pure full-quartic local objective assembled from expanded moments."""

from __future__ import annotations

import torch
from torch import Tensor

from .moments import ExpandedMoments, MomentContext


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
    ) -> None:
        if not isinstance(context, MomentContext):
            raise TypeError("context must be a MomentContext")
        if not isinstance(moments, ExpandedMoments):
            raise TypeError("moments must be ExpandedMoments")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
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

    @property
    def point_shape(self) -> tuple[int, int]:
        return self.context.geometry.point_shape

    def value(self, displacement: Tensor) -> Tensor:
        """Evaluate the complete quartic objective at ``displacement``."""

        self._validate_local(displacement)
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

    def _validate_local(self, value: Tensor) -> None:
        if value.shape != self.point_shape:
            raise ValueError(
                f"displacement must have shape {list(self.point_shape)}"
            )
        point = self.context.current_point
        if value.device != point.device or value.dtype != point.dtype:
            raise ValueError("displacement must match current point device and dtype")
