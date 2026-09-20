"""Representation-frame geometry shared by moment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from torchcst._runtime.validation import deferred, require

from .atoms import AtomDerivatives
from .quadratic import QuadraticFeatureGram, factored_quadratic_gram


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
            raise ValueError(f"linear must have shape {list(expected_linear_shape)}")
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
        return self.constant + torch.einsum("kpq,kq->kp", self.linear, displacement)

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
        device_solver: str = "jacobi",
    ) -> None:
        local_size = point_shape[0] * point_shape[1]
        if matrix.shape != (local_size, local_size):
            raise ValueError(f"matrix must have shape {[local_size, local_size]}")
        if damping < 0:
            raise ValueError("damping must be nonnegative")
        if rtol is not None and rtol < 0:
            raise ValueError("rtol must be nonnegative")
        if device_solver not in ("jacobi", "cholesky"):
            raise ValueError("device_solver must be jacobi or cholesky")
        if device_solver == "cholesky" and rtol is not None:
            raise ValueError("cholesky solves the damped system without an rtol cutoff")
        if device_solver == "cholesky" and damping <= 0:
            raise ValueError("cholesky requires explicit positive damping")
        self.device_solver = device_solver
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
        if self.device_solver == "cholesky":
            from ._cholesky import device_solve

            solution, valid = device_solve(matrix, right_hand_side.reshape(-1))
            require(
                valid,
                "damped Gram Cholesky solve failed validation",
                FloatingPointError,
            )
            return solution.reshape(self.point_shape)
        if deferred():
            from ._jacobi import device_pinv_solve

            solution, valid = device_pinv_solve(
                matrix, right_hand_side.reshape(-1), rtol=self.rtol
            )
            require(
                valid,
                "device Gram decomposition failed its residual check",
                FloatingPointError,
            )
            return solution.reshape(self.point_shape)
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

        return self.gram(frame, damping=damping, rtol=rtol).solve(pullback_numerator)

    def quadratic_feature_gram(
        self, *, point: Tensor, row_weight: Tensor, column_weight: Tensor, eps: float
    ) -> QuadraticFeatureGram | None:
        """Optional exact acceleration for a separable visible metric."""

        return None

    @property
    def supports_quadratic_feature_gram(self) -> bool:
        return False

    def local_quadratic_derivatives(
        self, point: Tensor
    ) -> tuple[Tensor, Tensor] | None:
        """Optional detached J[k,m,p], H[k,m,p,q] for exact solver models."""

        return None


class AutogradFrameGeometry(FrameGeometry):
    """Correctness implementation built from generic derivative contractions."""

    _MAX_DERIVATIVE_CACHE_ELEMENTS = 16_000_000

    def __init__(self, derivatives: AtomDerivatives) -> None:
        if not isinstance(derivatives, AtomDerivatives):
            raise TypeError("derivatives must be an AtomDerivatives instance")
        self.derivatives = derivatives
        self.device_solver = "jacobi"
        self._visible_shape = tuple(derivatives.represented().shape)
        self._cache_point: Tensor | None = None
        self._cache_jacobian: Tensor | None = None
        self._cache_hessian: Tensor | None = None
        self._cache_disabled = False

    @property
    def point_shape(self) -> tuple[int, int]:
        return self.derivatives.point_shape

    @property
    def visible_shape(self) -> tuple[int, ...]:
        return self._visible_shape

    def current_point(self) -> Tensor:
        return self.derivatives.current_point()

    @property
    def supports_quadratic_feature_gram(self) -> bool:
        return self.derivatives.factor_atoms is not None

    def local_quadratic_derivatives(
        self, point: Tensor
    ) -> tuple[Tensor, Tensor] | None:
        self._validate_local(point, name="quadratic derivative point")
        cached = self._local_derivatives(point)
        return None if cached is None else self._flattened_derivatives(cached)

    def quadratic_feature_gram(
        self, *, point: Tensor, row_weight: Tensor, column_weight: Tensor, eps: float
    ) -> QuadraticFeatureGram | None:
        self._validate_local(point, name="quadratic feature point")
        if self.derivatives.factor_atoms is None:
            return None
        return factored_quadratic_gram(
            self.derivatives.factor_atoms, point, row_weight, column_weight, eps
        )

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
        if deferred() and current_point.device.type == "cuda":
            from ._captured import call

            constant, linear = call(
                "frame_transport",
                self.derivatives._materialize_atoms,
                current_point,
                source_frame.point,
                source_frame.displacement,
                source_coefficients,
            )
            return AffinePullback(constant, linear)
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
        cached = self._local_derivatives(point)
        if cached is not None:
            jacobian, hessian = self._flattened_derivatives(cached)
            linear = torch.einsum("kmp,kp->m", jacobian, direction)
            quadratic = torch.einsum("kmpq,kp,kq->m", hessian, direction, direction)
            return (linear + 0.5 * quadratic).reshape(self.visible_shape)
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
        self._validate_visible(cotangent, name="cotangent")
        cached = self._local_derivatives(point)
        if cached is not None:
            jacobian, hessian = self._flattened_derivatives(cached)
            if displacement is not None:
                jacobian = jacobian + torch.einsum(
                    "kmpq,kq->kmp", hessian, displacement
                )
            return torch.einsum("kmp,m->kp", jacobian, cotangent.reshape(-1))
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

        cached = self._local_derivatives(frame.point)
        if cached is not None:
            jacobian, hessian = self._flattened_derivatives(cached)
            frame_jacobian = jacobian + torch.einsum(
                "kmpq,kq->kmp", hessian, frame.displacement
            )
            visible_columns = frame_jacobian.permute(1, 0, 2).reshape(-1, local_size)
            matrix = visible_columns.transpose(0, 1) @ visible_columns
            return GramSystem(
                matrix,
                self.point_shape,
                damping=damping,
                rtol=rtol,
                device_solver=self.device_solver,
            )

        def pushforward(direction: Tensor) -> Tensor:
            return self.derivatives.pushforward(
                direction,
                at=frame.displacement,
                parameter_point=frame.point,
            )

        visible_basis = torch.vmap(pushforward, chunk_size=64)(basis)
        visible_columns = visible_basis.flatten(start_dim=1).transpose(0, 1)
        matrix = visible_columns.transpose(0, 1) @ visible_columns
        return GramSystem(
            matrix,
            self.point_shape,
            damping=damping,
            rtol=rtol,
            device_solver=self.device_solver,
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

    def _validate_visible(self, value: Tensor, *, name: str) -> None:
        if value.shape != self.visible_shape:
            raise ValueError(f"{name} must have shape {list(self.visible_shape)}")
        parameter = self.derivatives.atoms.p
        if value.device != parameter.device or value.dtype != parameter.dtype:
            raise ValueError(f"{name} must match atom device and dtype")

    def _local_derivatives(self, point: Tensor) -> tuple[Tensor, Tensor] | None:
        if self._cache_disabled:
            return None
        if self._cache_point is point:
            assert self._cache_jacobian is not None
            assert self._cache_hessian is not None
            return self._cache_jacobian, self._cache_hessian
        if (
            not deferred()
            and self._cache_point is not None
            and torch.equal(self._cache_point, point)
        ):
            assert self._cache_jacobian is not None
            assert self._cache_hessian is not None
            return self._cache_jacobian, self._cache_hessian

        atoms, parameters = self.point_shape
        visible_size = self.derivatives.represented(point).numel()
        cache_elements = atoms * visible_size * (parameters + parameters**2)
        if cache_elements > self._MAX_DERIVATIVE_CACHE_ELEMENTS:
            self._cache_disabled = True
            return None

        if deferred() and point.device.type == "cuda":
            from ._captured import call

            jacobian, hessian = call(
                "local_derivatives", self.derivatives._materialize_atoms, point
            )
        else:
            jacobian, hessian = self.derivatives.materialized_local_derivatives(
                parameter_point=point
            )
        self._cache_point = point
        self._cache_jacobian = jacobian
        self._cache_hessian = hessian
        return jacobian, hessian

    def _flattened_derivatives(
        self, cached: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        jacobian, hessian = cached
        atoms, parameters = self.point_shape
        return (
            jacobian.reshape(atoms, -1, parameters),
            hessian.reshape(atoms, -1, parameters, parameters),
        )
