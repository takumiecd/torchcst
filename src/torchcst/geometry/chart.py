"""Fixed-cardinality observation charts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from .geometry import EuclideanGeometry, Geometry, SphereGeometry


class Chart(nn.Module, ABC):
    """Geometry-backed observation sites for a logical tensor shape."""

    def get_extra_state(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "chart_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "features": self.features,
            "embedding_dim": self.embedding_dim,
            "trainable": self.trainable,
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("chart checkpoint contract differs from this chart")

    @classmethod
    def points(
        cls,
        coordinates: Tensor,
        *,
        geometry: Geometry | None = None,
        trainable: bool = False,
    ) -> Chart:
        return ExplicitChart.points(coordinates, geometry=geometry, trainable=trainable)

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
        return ExplicitChart.linspace(
            size,
            spacing=spacing,
            low=low,
            high=high,
            center=center,
            trainable=trainable,
        )

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
        return ExplicitChart.grid(
            shape,
            spacing=spacing,
            low=low,
            high=high,
            center=center,
            trainable=trainable,
        )

    @classmethod
    def sphere(
        cls,
        features: int,
        *,
        intrinsic_dim: int,
        radius: float = 1.0,
        representation: Literal["ambient", "intrinsic"] = "ambient",
        chart_margin: float = 0.05,
        trainable: bool = False,
    ) -> Chart:
        return ExplicitChart.sphere(
            features,
            intrinsic_dim=intrinsic_dim,
            radius=radius,
            representation=representation,
            chart_margin=chart_margin,
            trainable=trainable,
        )

    @property
    @abstractmethod
    def features(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.features,)

    @property
    @abstractmethod
    def reference(self) -> Tensor: ...

    @property
    def embedding_dim(self) -> int:
        return self.geometry.embedding_dim

    @property
    def intrinsic_dim(self) -> int:
        return self.geometry.intrinsic_dim

    @property
    def center_parameter_dim(self) -> int:
        return self.geometry.center_parameter_dim

    @property
    @abstractmethod
    def trainable(self) -> bool: ...

    @abstractmethod
    def positions(self, indices: Tensor) -> Tensor: ...

    @abstractmethod
    def squared_distance(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor: ...

    @abstractmethod
    def center_offsets(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor: ...

    @abstractmethod
    def initialize_centers(self, atoms: int, *, mode: str) -> Tensor: ...


class ExplicitChart(Chart):
    """A fixed-cardinality collection of observation coordinates.

    Cartesian constructors take a grid step ``spacing`` in the same units as
    kernel bandwidth. The chart is centered at ``center`` (default 0), so
    ``n`` points occupy ``[center - (n-1)*spacing/2, center + (n-1)*spacing/2]``.
    ``low``/``high`` is the alternate inclusive-endpoint form; do not mix it
    with ``spacing``. There is no implicit ``[-1, 1]`` domain.
    """

    def __init__(
        self,
        coordinates: Tensor,
        *,
        geometry: Geometry | None = None,
        trainable: bool = False,
    ) -> None:
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

        if geometry is None:
            geometry = EuclideanGeometry(coordinates.shape[1])
        elif not isinstance(geometry, Geometry):
            raise TypeError("geometry must be a Geometry")
        if coordinates.shape[1] != geometry.embedding_dim:
            raise ValueError(
                "coordinate width must match geometry.embedding_dim "
                f"({geometry.embedding_dim})"
            )
        geometry.validate_points(coordinates, name="coordinates")
        self.geometry = geometry

        owned = coordinates.detach().clone()
        if trainable:
            self.coordinates = nn.Parameter(owned)
        else:
            self.register_buffer("coordinates", owned)

    def get_extra_state(self) -> dict[str, object]:
        """Record fixed chart properties outside the coordinate tensor."""

        return {
            "format_version": 1,
            # Preserve the checkpoint tag of the original concrete Chart.
            "chart_type": "torchcst.geometry.chart.Chart",
            "features": self.features,
            "embedding_dim": self.embedding_dim,
            "trainable": self.trainable,
        }

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("chart checkpoint contract differs from this chart")

    @classmethod
    def points(
        cls,
        coordinates: Tensor,
        *,
        geometry: Geometry | None = None,
        trainable: bool = False,
    ) -> Chart:
        """Construct a chart from an explicit ``[features, dimensions]`` tensor."""

        return cls(coordinates, geometry=geometry, trainable=trainable)

    @classmethod
    def sphere(
        cls,
        features: int,
        *,
        intrinsic_dim: int,
        radius: float = 1.0,
        representation: Literal["ambient", "intrinsic"] = "ambient",
        chart_margin: float = 0.05,
        trainable: bool = False,
    ) -> Chart:
        """Construct points sampled uniformly on the intrinsic sphere ``S^d``."""

        cls._validate_size(features, name="features")
        geometry = SphereGeometry(
            intrinsic_dim,
            radius=radius,
            representation=representation,
            chart_margin=chart_margin,
        )
        coordinates = geometry.sample_sites(features)
        return cls(coordinates, geometry=geometry, trainable=trainable)

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
                return tuple(
                    cls._positive_float(item, name=f"{name}[{i}]")
                    for i, item in enumerate(value)
                )
            return tuple(
                cls._finite_float(item, name=f"{name}[{i}]")
                for i, item in enumerate(value)
            )
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
    def shape(self) -> tuple[int, ...]:
        """Logical site shape; explicit charts are one-dimensional."""

        return (self.features,)

    @property
    def reference(self) -> Tensor:
        return self.coordinates

    @property
    def device(self) -> torch.device:
        return self.coordinates.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coordinates.dtype

    def positions(self, indices: Tensor) -> Tensor:
        """Return coordinates for requested flattened site indices."""

        return self.coordinates.index_select(0, indices)

    @property
    def dim(self) -> int:
        """Stored coordinate width; retained as an alias for ``embedding_dim``."""

        return self.coordinates.shape[1]

    @property
    def embedding_dim(self) -> int:
        """Stored coordinate width of each observation point."""

        return self.geometry.embedding_dim

    @property
    def intrinsic_dim(self) -> int:
        """Geometric degrees of freedom of one point."""

        return self.geometry.intrinsic_dim

    @property
    def center_parameter_dim(self) -> int:
        """Stored coordinate width of one atom center."""

        return self.geometry.center_parameter_dim

    def squared_distance(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor:
        """Pairwise site-center squared distance in this chart's geometry."""

        sites = self.coordinates if selection is None else self.coordinates[selection]
        return self.geometry.squared_distance(sites, centers)

    def center_offsets(
        self, centers: Tensor, selection: slice | Tensor | None = None
    ) -> Tensor:
        """Site-center offsets in each center's tangent space."""

        sites = self.coordinates if selection is None else self.coordinates[selection]
        return self.geometry.center_offsets(sites, centers)

    def initialize_centers(self, atoms: int, *, mode: str) -> Tensor:
        """Initialize atom centers in the chart's geometry."""

        return self.geometry.initialize_centers(self.coordinates, atoms, mode=mode)

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
            f"intrinsic_dim={self.intrinsic_dim}",
            f"embedding_dim={self.embedding_dim}",
            f"center_parameter_dim={self.center_parameter_dim}",
            f"trainable={self.trainable}",
        ]
        spacing = self.spacing
        if spacing is not None:
            if spacing.numel() == 1:
                parts.insert(2, f"spacing={float(spacing)}")
            else:
                parts.insert(2, f"spacing={tuple(float(s) for s in spacing)}")
        return ", ".join(parts)
