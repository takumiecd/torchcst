"""Fixed-cardinality observation charts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Chart(nn.Module):
    """A fixed-cardinality collection of observation coordinates.

    Cartesian constructors take a grid step ``spacing`` in the same units as
    kernel bandwidth. The chart is centered at ``center`` (default 0), so
    ``n`` points occupy ``[center - (n-1)*spacing/2, center + (n-1)*spacing/2]``.
    ``low``/``high`` is the alternate inclusive-endpoint form; do not mix it
    with ``spacing``. There is no implicit ``[-1, 1]`` domain.
    """

    def __init__(self, coordinates: Tensor, *, trainable: bool = False) -> None:
        super().__init__()
        if not isinstance(coordinates, Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.ndim != 2:
            raise ValueError("coordinates must have shape [features, dimensions]")
        if coordinates.shape[0] < 1 or coordinates.shape[1] < 1:
            raise ValueError("a chart must contain at least one point and dimension")
        if not coordinates.is_floating_point():
            raise TypeError("chart coordinates must have a floating-point dtype")
        if not torch.isfinite(coordinates).all():
            raise ValueError("chart coordinates must be finite")

        owned = coordinates.detach().clone()
        if trainable:
            self.coordinates = nn.Parameter(owned)
        else:
            self.register_buffer("coordinates", owned)

    @classmethod
    def points(cls, coordinates: Tensor, *, trainable: bool = False) -> Chart:
        """Construct a chart from an explicit ``[features, dimensions]`` tensor."""

        return cls(coordinates, trainable=trainable)

    @classmethod
    def linspace(
        cls,
        size: int,
        *,
        spacing: float | None = None,
        low: float | None = None,
        high: float | None = None,
        center: float | None = None,
        trainable: bool = False,
    ) -> Chart:
        """Construct a one-dimensional evenly spaced chart."""

        cls._validate_size(size, name="size")
        axis, step = cls._axis(
            size,
            spacing=spacing,
            low=low,
            high=high,
            center=center,
            name="linspace",
        )
        chart = cls(axis.unsqueeze(-1), trainable=trainable)
        chart._set_spacing((step,) if step is not None else None)
        return chart

    @classmethod
    def grid(
        cls,
        shape: Sequence[int],
        *,
        spacing: float | Sequence[float] | None = None,
        low: float | None = None,
        high: float | None = None,
        center: float | Sequence[float] | None = None,
        trainable: bool = False,
    ) -> Chart:
        """Construct a Cartesian grid with one coordinate axis per shape entry."""

        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise TypeError("shape must be a non-empty sequence of integers")
        shape = tuple(shape)
        if not shape:
            raise ValueError("shape must contain at least one dimension")
        for index, size in enumerate(shape):
            cls._validate_size(size, name=f"shape[{index}]")

        dim = len(shape)
        steps = cls._per_axis_values(spacing, dim=dim, name="spacing", allow_none=True)
        centers = cls._per_axis_values(center, dim=dim, name="center", allow_none=True)
        if steps is not None and (low is not None or high is not None):
            raise ValueError("specify spacing or low/high, not both")
        if (low is None) ^ (high is None):
            raise ValueError("low and high must be passed together")
        if steps is None and low is None:
            raise ValueError("specify spacing or low and high")
        if low is not None and center is not None:
            raise ValueError("center is only used with spacing")

        axes = []
        stored_steps: list[float] = []
        for index, size in enumerate(shape):
            axis, step = cls._axis(
                size,
                spacing=None if steps is None else steps[index],
                low=low,
                high=high,
                center=None if centers is None else centers[index],
                name=f"grid[{index}]",
            )
            axes.append(axis)
            if step is not None:
                stored_steps.append(step)
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
            -1, dim
        )
        chart = cls(coordinates, trainable=trainable)
        chart._set_spacing(tuple(stored_steps) if len(stored_steps) == dim else None)
        return chart

    @staticmethod
    def _validate_size(size: int, *, name: str) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"{name} must be an integer")
        if size < 1:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _positive_float(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    @staticmethod
    def _finite_float(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    @classmethod
    def _per_axis_values(
        cls,
        value: float | Sequence[float] | None,
        *,
        dim: int,
        name: str,
        allow_none: bool,
    ) -> tuple[float, ...] | None:
        if value is None:
            if allow_none:
                return None
            raise TypeError(f"{name} is required")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != dim:
                raise ValueError(f"{name} must have one entry per axis")
            if name == "spacing":
                return tuple(cls._positive_float(item, name=f"{name}[{i}]") for i, item in enumerate(value))
            return tuple(cls._finite_float(item, name=f"{name}[{i}]") for i, item in enumerate(value))
        if name == "spacing":
            number = cls._positive_float(value, name=name)
        else:
            number = cls._finite_float(value, name=name)
        return (number,) * dim

    @classmethod
    def _axis(
        cls,
        size: int,
        *,
        spacing: float | None,
        low: float | None,
        high: float | None,
        center: float | None,
        name: str,
    ) -> tuple[Tensor, float | None]:
        if spacing is not None and (low is not None or high is not None):
            raise ValueError("specify spacing or low/high, not both")
        if (low is None) ^ (high is None):
            raise ValueError("low and high must be passed together")
        if spacing is None and low is None:
            raise ValueError("specify spacing or low and high")
        if spacing is not None:
            step = cls._positive_float(spacing, name="spacing")
            origin = 0.0 if center is None else cls._finite_float(center, name="center")
            index = torch.arange(size, dtype=torch.get_default_dtype())
            axis = (index - (size - 1) / 2) * step + origin
            return axis, step
        if center is not None:
            raise ValueError("center is only used with spacing")
        start = cls._finite_float(low, name="low")
        end = cls._finite_float(high, name="high")
        if size == 1:
            return torch.tensor([start], dtype=torch.get_default_dtype()), None
        if end < start:
            raise ValueError("high must not be smaller than low")
        axis = torch.linspace(start, end, size)
        return axis, float(axis[1] - axis[0])

    def _set_spacing(self, steps: tuple[float, ...] | None) -> None:
        if steps is None:
            return
        self.register_buffer(
            "_chart_spacing",
            torch.tensor(steps, dtype=self.coordinates.dtype),
        )

    @property
    def features(self) -> int:
        """Number of observation points."""

        return self.coordinates.shape[0]

    @property
    def dim(self) -> int:
        """Coordinate dimension of each observation point."""

        return self.coordinates.shape[1]

    @property
    def spacing(self) -> Tensor | None:
        """Grid step per axis, or ``None`` when the chart is not a regular grid."""

        return getattr(self, "_chart_spacing", None)

    @property
    def trainable(self) -> bool:
        """Whether the coordinates participate in gradient-based training."""

        return isinstance(self.coordinates, nn.Parameter)

    def extra_repr(self) -> str:
        parts = [
            f"features={self.features}",
            f"dim={self.dim}",
            f"trainable={self.trainable}",
        ]
        spacing = self.spacing
        if spacing is not None:
            if spacing.numel() == 1:
                parts.insert(2, f"spacing={float(spacing)}")
            else:
                parts.insert(2, f"spacing={tuple(float(s) for s in spacing)}")
        return ", ".join(parts)
