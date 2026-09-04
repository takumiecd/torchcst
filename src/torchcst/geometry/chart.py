"""Fixed-cardinality observation charts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Chart(nn.Module):
    """A fixed-cardinality collection of observation coordinates."""

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
        low: float = -1.0,
        high: float = 1.0,
        trainable: bool = False,
    ) -> Chart:
        """Construct a one-dimensional evenly spaced chart."""

        cls._validate_size(size, name="size")
        coordinates = torch.linspace(low, high, size).unsqueeze(-1)
        return cls(coordinates, trainable=trainable)

    @classmethod
    def grid(
        cls,
        shape: Sequence[int],
        *,
        low: float = -1.0,
        high: float = 1.0,
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

        axes = [torch.linspace(low, high, size) for size in shape]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), dim=-1
        ).reshape(-1, len(shape))
        return cls(coordinates, trainable=trainable)

    @staticmethod
    def _validate_size(size: int, *, name: str) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"{name} must be an integer")
        if size < 1:
            raise ValueError(f"{name} must be positive")

    @property
    def features(self) -> int:
        """Number of observation points."""

        return self.coordinates.shape[0]

    @property
    def dim(self) -> int:
        """Coordinate dimension of each observation point."""

        return self.coordinates.shape[1]

    @property
    def trainable(self) -> bool:
        """Whether the coordinates participate in gradient-based training."""

        return isinstance(self.coordinates, nn.Parameter)

    def extra_repr(self) -> str:
        return (
            f"features={self.features}, dim={self.dim}, "
            f"trainable={self.trainable}"
        )
