"""Representation-frame geometry shared by moment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from .atoms import AtomDerivatives


@dataclass(frozen=True)
class RepresentationFrame:
    """The visible tangent frame ``V_point(displacement)``."""

    point: Tensor
    displacement: Tensor


@dataclass(frozen=True)
class AffinePullback:
    """An atom-local affine pullback ``constant + linear @ d``."""

    constant: Tensor
    linear: Tensor

    def __post_init__(self) -> None:
        if self.constant.ndim != 2:
            raise ValueError("constant must have shape [K, P]")
        expected_linear_shape = (*self.constant.shape, self.constant.shape[1])
        if self.linear.shape != expected_linear_shape:
            raise ValueError(
                f"linear must have shape {list(expected_linear_shape)}"
            )
        if (
            self.constant.device != self.linear.device
            or self.constant.dtype != self.linear.dtype
        ):
            raise ValueError("constant and linear must share one device and dtype")

    def at(self, displacement: Tensor) -> Tensor:
        """Evaluate the pullback at one local displacement."""

        if displacement.shape != self.constant.shape:
            raise ValueError(
                f"displacement must have shape {list(self.constant.shape)}"
            )
        if (
            displacement.device != self.constant.device
            or displacement.dtype != self.constant.dtype
        ):
            raise ValueError("displacement must match pullback device and dtype")
        return self.constant + torch.einsum(
            "kpq,kq->kp", self.linear, displacement
        )

    def scaled(self, factor: float | Tensor) -> AffinePullback:
        return AffinePullback(self.constant * factor, self.linear * factor)

    def __add__(self, other: AffinePullback) -> AffinePullback:
        if not isinstance(other, AffinePullback):
            return NotImplemented
        return AffinePullback(
            self.constant + other.constant,
            self.linear + other.linear,
        )


class GramSystem:
    """A local Gram system with a replaceable solve implementation."""

    def __init__(
        self,
        matrix: Tensor,
        point_shape: tuple[int, int],
        *,
        damping: float = 0.0,
        rtol: float | None = None,
    ) -> None:
        local_size = point_shape[0] * point_shape[1]
        if matrix.shape != (local_size, local_size):
            raise ValueError(
                f"matrix must have shape {[local_size, local_size]}"
            )
        if damping < 0:
            raise ValueError("damping must be nonnegative")
        if rtol is not None and rtol < 0:
            raise ValueError("rtol must be nonnegative")
        self._matrix = matrix.detach().clone()
        self.point_shape = point_shape
        self.damping = float(damping)
        self.rtol = rtol

    @property
    def matrix(self) -> Tensor:
        """Return the dense local correctness matrix as an isolated snapshot."""

        return self._matrix.clone()

    def matvec(self, value: Tensor) -> Tensor:
        self._validate_local(value)
        return (self._matrix @ value.reshape(-1)).reshape(self.point_shape)

    def solve(self, right_hand_side: Tensor) -> Tensor:
        """Return the minimum-norm solution of the optionally damped system."""

        self._validate_local(right_hand_side)
        matrix = self._matrix
        if self.damping:
            matrix = matrix + self.damping * torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            )
        options = {} if self.rtol is None else {"rtol": self.rtol}
        inverse = torch.linalg.pinv(matrix, **options)
        return (inverse @ right_hand_side.reshape(-1)).reshape(self.point_shape)

    def _validate_local(self, value: Tensor) -> None:
        if value.shape != self.point_shape:
            raise ValueError(f"value must have shape {list(self.point_shape)}")
        if value.device != self._matrix.device or value.dtype != self._matrix.dtype:
            raise ValueError("value must match Gram system device and dtype")


class FrameGeometry(ABC):
    """Optimizer-independent contractions between CST representation frames."""

    @property
    @abstractmethod
    def point_shape(self) -> tuple[int, int]: ...

    @property
    @abstractmethod
    def visible_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def current_point(self) -> Tensor: ...

    @abstractmethod
    def frame(
        self, point: Tensor, displacement: Tensor | None = None
    ) -> RepresentationFrame: ...

    @abstractmethod
    def pullback_from_frame(
        self,
        *,
        current_point: Tensor,
        source_frame: RepresentationFrame,
        source_coefficients: Tensor,
    ) -> AffinePullback: ...

    @abstractmethod
    def displacement(self, direction: Tensor, *, point: Tensor) -> Tensor:
        """Evaluate the second-order represented displacement."""

    @abstractmethod
    def pullback(
        self,
        cotangent: Tensor,
        *,
        point: Tensor,
        displacement: Tensor | None = None,
    ) -> Tensor:
        """Evaluate a represented cotangent in one local frame."""

    @abstractmethod
    def gram(
        self,
        frame: RepresentationFrame,
        *,
        damping: float = 0.0,
        rtol: float | None = None,
    ) -> GramSystem: ...

    def compress(
        self,
        *,
        frame: RepresentationFrame,
        pullback_numerator: Tensor,
        damping: float = 0.0,
        rtol: float | None = None,
    ) -> Tensor:
        """Compress a visible target from its frame pullback numerator."""

        return self.gram(frame, damping=damping, rtol=rtol).solve(
            pullback_numerator
        )


class AutogradFrameGeometry(FrameGeometry):
    """Correctness implementation built from generic derivative contractions."""

    def __init__(self, derivatives: AtomDerivatives) -> None:
        if not isinstance(derivatives, AtomDerivatives):
            raise TypeError("derivatives must be an AtomDerivatives instance")
        self.derivatives = derivatives

    @property
    def point_shape(self) -> tuple[int, int]:
        return self.derivatives.point_shape

    @property
    def visible_shape(self) -> tuple[int, ...]:
        return tuple(self.derivatives.represented().shape)

    def current_point(self) -> Tensor:
        return self.derivatives.current_point()

    def frame(
        self, point: Tensor, displacement: Tensor | None = None
    ) -> RepresentationFrame:
        self._validate_local(point, name="frame point")
        if displacement is None:
            displacement = torch.zeros_like(point)
        self._validate_local(displacement, name="frame displacement")
        return RepresentationFrame(
            point=point.detach().clone(),
            displacement=displacement.detach().clone(),
        )

    def pullback_from_frame(
        self,
        *,
        current_point: Tensor,
        source_frame: RepresentationFrame,
        source_coefficients: Tensor,
    ) -> AffinePullback:
        self._validate_local(current_point, name="current point")
        self._validate_frame(source_frame)
        self._validate_local(source_coefficients, name="source coefficients")
        visible_tangent = self.derivatives.pushforward(
            source_coefficients,
            at=source_frame.displacement,
            parameter_point=source_frame.point,
        )
        constant = self.derivatives.pullback(
            visible_tangent,
            parameter_point=current_point,
        )
        linear = self.derivatives.contracted_hessian(
            visible_tangent,
            parameter_point=current_point,
        )
        return AffinePullback(constant.detach(), linear.detach())

    def displacement(self, direction: Tensor, *, point: Tensor) -> Tensor:
        self._validate_local(point, name="displacement point")
        self._validate_local(direction, name="direction")
        return self.derivatives.displacement(direction, parameter_point=point)

    def pullback(
        self,
        cotangent: Tensor,
        *,
        point: Tensor,
        displacement: Tensor | None = None,
    ) -> Tensor:
        self._validate_local(point, name="pullback point")
        if displacement is not None:
            self._validate_local(displacement, name="pullback displacement")
        return self.derivatives.pullback(
            cotangent,
            at=displacement,
            parameter_point=point,
        )

    def gram(
        self,
        frame: RepresentationFrame,
        *,
        damping: float = 0.0,
        rtol: float | None = None,
    ) -> GramSystem:
        self._validate_frame(frame)
        local_size = self.point_shape[0] * self.point_shape[1]
        basis = torch.eye(
            local_size,
            device=frame.point.device,
            dtype=frame.point.dtype,
        ).reshape(local_size, *self.point_shape)
        columns = []
        for direction in basis:
            visible = self.derivatives.pushforward(
                direction,
                at=frame.displacement,
                parameter_point=frame.point,
            )
            columns.append(visible.reshape(-1))
        visible_columns = torch.stack(columns, dim=1)
        matrix = visible_columns.transpose(0, 1) @ visible_columns
        return GramSystem(
            matrix,
            self.point_shape,
            damping=damping,
            rtol=rtol,
        )

    def visible_pushforward(
        self, frame: RepresentationFrame, coefficients: Tensor
    ) -> Tensor:
        """Reference-only expansion used by correctness tests."""

        self._validate_frame(frame)
        self._validate_local(coefficients, name="coefficients")
        return self.derivatives.pushforward(
            coefficients,
            at=frame.displacement,
            parameter_point=frame.point,
        )

    def _validate_frame(self, frame: RepresentationFrame) -> None:
        if not isinstance(frame, RepresentationFrame):
            raise TypeError("frame must be a RepresentationFrame")
        self._validate_local(frame.point, name="frame point")
        self._validate_local(frame.displacement, name="frame displacement")

    def _validate_local(self, value: Tensor, *, name: str) -> None:
        if value.shape != self.point_shape:
            raise ValueError(f"{name} must have shape {list(self.point_shape)}")
        parameter = self.derivatives.atoms.p
        if value.device != parameter.device or value.dtype != parameter.dtype:
            raise ValueError(f"{name} must match atom device and dtype")
