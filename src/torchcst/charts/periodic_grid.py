"""Implicit equally spaced periodic lattice for a single linear operator."""

import math
from dataclasses import dataclass, field, replace

import torch

from torchcst._validation import _shape
from torchcst.geometry.spec import FlatTorusGeometrySpec

from .base import ChartSpec, ChartState


@dataclass(frozen=True, kw_only=True)
class PeriodicGridChartSpec(ChartSpec):
    grid_shape: tuple[int, ...]
    output_dims: int
    origin: tuple[float, ...]
    kind: str = field(default="periodic_grid", init=False)

    def __post_init__(self):
        super().__post_init__()
        _shape(self.grid_shape)
        dim = len(self.grid_shape)
        if (
            type(self.geometry) is not FlatTorusGeometrySpec
            or self.geometry.intrinsic_dim != dim
        ):
            raise ValueError("periodic_grid requires a matching FlatTorus geometry")
        if type(self.output_dims) is not int or not 0 < self.output_dims < dim:
            raise ValueError(
                "output_dims must split nonempty output and input coordinates"
            )
        if self.shape != (
            math.prod(self.grid_shape[: self.output_dims]),
            math.prod(self.grid_shape[self.output_dims :]),
        ):
            raise ValueError("chart shape must match the flattened output/input grid")
        if (
            not isinstance(self.origin, tuple)
            or len(self.origin) != dim
            or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in self.origin
            )
        ):
            raise ValueError(
                "origin must be a finite immutable tuple matching grid_shape"
            )


class PeriodicGridChartState(ChartState):
    """Own only O(D) coordinate data, regardless of the number of lattice sites."""

    spec_type = PeriodicGridChartSpec

    def __init__(self, spec, *, device=None, dtype=None):
        super().__init__(spec, device=device, dtype=dtype)
        self.grid_shape = spec.grid_shape
        self.output_dims = spec.output_dims
        self.register_buffer(
            "origin",
            torch.tensor(spec.origin, device=device, dtype=self.geometry.periods.dtype),
        )
        self.declaration()  # Reject buffers that overflow/underflow in the chosen dtype.

    @property
    def reference(self):
        return self.origin

    @property
    def spacing(self):
        return torch.stack(
            [
                period / size
                for period, size in zip(self.geometry.periods, self.grid_shape)
            ]
        )

    def declaration(self):
        return replace(
            self.spec,
            geometry=self.geometry.declaration(),
            origin=tuple(self.origin.detach().cpu().tolist()),
        )

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "grid_shape": self.grid_shape,
            "output_dims": self.output_dims,
        }
