"""Implicit regular grids grouped by logical weight axis."""

import math
from dataclasses import dataclass, field, replace

import torch

from torchcst._coordinate_state import _floating_dtype
from torchcst._validation import _shape
from torchcst.geometry.spec import EuclideanGeometrySpec, FlatTorusGeometrySpec

from .base import ChartSpec, ChartState


@dataclass(frozen=True, kw_only=True)
class RegularGridChartSpec(ChartSpec):
    grid_shape: tuple[tuple[int, ...], ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    shape: tuple[int, ...] = field(init=False)
    kind: str = field(default="regular_grid", init=False)

    def __post_init__(self):
        if not isinstance(self.grid_shape, tuple) or not self.grid_shape:
            raise ValueError("grid_shape needs a nonempty tuple per logical axis")
        for shape in self.grid_shape:
            _shape(shape)
        object.__setattr__(self, "shape", tuple(map(math.prod, self.grid_shape)))
        super().__post_init__()
        dim = len(self.coordinate_shape)
        if (
            type(self.geometry) not in (EuclideanGeometrySpec, FlatTorusGeometrySpec)
            or self.geometry.intrinsic_dim != dim
        ):
            raise ValueError(
                "regular_grid requires a matching Euclidean or FlatTorus geometry"
            )
        for name in ("spacing", "origin"):
            value = getattr(self, name)
            if (
                not isinstance(value, tuple)
                or len(value) != dim
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in value
                )
                or (name == "spacing" and any(v <= 0 for v in value))
            ):
                raise ValueError(
                    f"{name} must be a finite immutable tuple matching coordinates"
                )

    @property
    def coordinate_shape(self):
        return tuple(n for group in self.grid_shape for n in group)


class RegularGridChartState(ChartState):
    """Own O(D) origins/spacings, with no site coordinate or index table."""

    spec_type = RegularGridChartSpec

    def __init__(self, spec, *, device=None, dtype=None):
        dtype = _floating_dtype(dtype)
        super().__init__(spec, device=device, dtype=dtype)
        self.grid_shape = spec.grid_shape
        self.coordinate_shape = spec.coordinate_shape
        self.register_buffer(
            "origin", torch.tensor(spec.origin, device=device, dtype=dtype)
        )
        self.register_buffer(
            "spacing",
            torch.tensor(spec.spacing, device=device, dtype=self.origin.dtype),
        )
        self.declaration()

    @property
    def reference(self):
        return self.origin

    def declaration(self):
        return replace(
            self.spec,
            geometry=self.geometry.declaration(),
            origin=tuple(self.origin.detach().cpu().tolist()),
            spacing=tuple(self.spacing.detach().cpu().tolist()),
        )

    def get_extra_state(self):
        return {**super().get_extra_state(), "grid_shape": self.grid_shape}
