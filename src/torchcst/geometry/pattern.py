"""Small, indexable site patterns; Cartesian positions are never stored."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class SitePattern(nn.Module, ABC):
    """Map flattened site indices to coordinates in a local geometry."""

    def get_extra_state(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "pattern_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "features": self.features,
            "dim": self.dim,
            "layout": self._layout(),
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("site pattern checkpoint contract differs")

    @abstractmethod
    def _layout(self) -> tuple: ...

    @property
    @abstractmethod
    def features(self) -> int: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def reference(self) -> Tensor: ...

    @abstractmethod
    def positions(self, indices: Tensor) -> Tensor: ...

    @abstractmethod
    def bounds(self) -> tuple[Tensor, Tensor]: ...


class GridPattern(SitePattern):
    """Row-major grid specified by spacing and center, or inclusive endpoints."""

    def __init__(
        self,
        shape: Sequence[int],
        *,
        spacing: float | Sequence[float] | None = None,
        center: float | Sequence[float] | None = None,
        low: float | Sequence[float] | None = None,
        high: float | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        if not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("shape must contain positive integers")
        dim = len(self.shape)

        def values(value, name):
            entries = value if isinstance(value, Sequence) else (value,) * dim
            if len(entries) != dim:
                raise ValueError(f"{name} must have one value per axis")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in entries
            ):
                raise TypeError(f"{name} must contain real numbers")
            result = tuple(float(v) for v in entries)
            if not all(math.isfinite(v) for v in result):
                raise ValueError(f"{name} must be finite")
            return result

        if (low is None) != (high is None):
            raise ValueError("low and high must be given together")
        if (spacing is None) == (low is None):
            raise ValueError("specify spacing or low/high")
        if low is not None and center is not None:
            raise ValueError("center is only used with spacing")
        if spacing is not None:
            steps = values(spacing, "spacing")
            if any(step <= 0 for step in steps):
                raise ValueError("spacing must be positive")
            centers = values(0.0 if center is None else center, "center")
            starts = tuple(
                c - (n - 1) * s / 2 for c, n, s in zip(centers, self.shape, steps)
            )
        else:
            starts = values(low, "low")
            ends = values(high, "high")
            if any(b < a for a, b in zip(starts, ends)):
                raise ValueError("high must not be smaller than low")
            steps = tuple(
                (b - a) / (n - 1) if n > 1 else 0.0
                for a, b, n in zip(starts, ends, self.shape)
            )
        self.register_buffer(
            "start", torch.tensor(starts, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "spacing", torch.tensor(steps, dtype=torch.get_default_dtype())
        )

    @property
    def features(self) -> int:
        return math.prod(self.shape)

    def _layout(self) -> tuple:
        return self.shape

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def reference(self) -> Tensor:
        return self.start

    def positions(self, indices: Tensor) -> Tensor:
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional long tensor")
        if indices.device != self.start.device:
            raise ValueError("indices and pattern must be on the same device")
        if bool(((indices < 0) | (indices >= self.features)).any()):
            raise IndexError("site index out of bounds")
        remainder = indices
        axes = []
        for size in reversed(self.shape):
            axes.append(remainder % size)
            remainder = torch.div(remainder, size, rounding_mode="floor")
        axes.reverse()
        return (
            torch.stack(axes, dim=-1).to(self.start.dtype) * self.spacing + self.start
        )

    def bounds(self) -> tuple[Tensor, Tensor]:
        end = self.start + self.spacing * self.start.new_tensor(
            [n - 1 for n in self.shape]
        )
        return self.start, end


class LinePattern(GridPattern):
    def __init__(
        self,
        size: int,
        *,
        spacing: float | None = None,
        center: float | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        super().__init__((size,), spacing=spacing, center=center, low=low, high=high)


class PointsPattern(SitePattern):
    """An explicit small site table, useful for irregular local layouts."""

    def __init__(self, coordinates: Tensor) -> None:
        super().__init__()
        if coordinates.ndim != 2 or min(coordinates.shape) < 1:
            raise ValueError("coordinates must have shape [sites, dimensions]")
        if not coordinates.is_floating_point() or not bool(
            torch.isfinite(coordinates).all()
        ):
            raise ValueError("coordinates must be finite floating-point values")
        self.register_buffer("coordinates", coordinates.detach().clone())

    @property
    def features(self) -> int:
        return self.coordinates.shape[0]

    def _layout(self) -> tuple:
        return tuple(self.coordinates.shape)

    @property
    def dim(self) -> int:
        return self.coordinates.shape[1]

    @property
    def reference(self) -> Tensor:
        return self.coordinates

    def positions(self, indices: Tensor) -> Tensor:
        return self.coordinates.index_select(0, indices)

    def bounds(self) -> tuple[Tensor, Tensor]:
        return self.coordinates.amin(0), self.coordinates.amax(0)
