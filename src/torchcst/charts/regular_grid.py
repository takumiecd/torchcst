"""Implicit regular grids grouped by logical weight axis."""

import math
from dataclasses import dataclass, field, replace

from torchcst._coordinate_state import Placement
from torchcst._validation import _shape
from torchcst.geometry.spec import EuclideanGeometrySpec, FlatTorusGeometrySpec

from .base import ChartSpec, ChartState


@dataclass(frozen=True, kw_only=True)
class RegularGridChartSpec(ChartSpec):
    grid_shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    shape: tuple[int, ...] = field(init=False)
    kind: str = field(default="regular_grid", init=False)

    def __post_init__(self):
        _shape(self.grid_shape)
        object.__setattr__(self, "shape", self.grid_shape)
        super().__post_init__()
        dim = len(self.grid_shape)
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
        return self.grid_shape


class RegularGridChartState(ChartState):
    """Own O(D) immutable grid metadata, without Tensor buffers or site tables."""

    spec_type = RegularGridChartSpec

    def __init__(self, spec, *, device=None, dtype=None):
        self._placement = Placement(device=device, dtype=dtype)
        super().__init__(spec, device=device, dtype=self._placement.dtype)
        self.grid_shape = spec.grid_shape
        self.coordinate_shape = spec.coordinate_shape

    @property
    def origin(self):
        return self.spec.origin

    @property
    def spacing(self):
        return self.spec.spacing

    @property
    def reference(self):
        return self._placement

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._placement = self._placement.converted(fn)
        return result

    def declaration(self):
        return replace(self.spec, geometry=self.geometry.declaration())

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "format_version": 3,
            "grid_shape": self.grid_shape,
            "origin": self.spec.origin,
            "spacing": self.spec.spacing,
        }


from torchcst._builtin_callbacks import seal_functions as _seal_functions

_seal_functions(globals())
