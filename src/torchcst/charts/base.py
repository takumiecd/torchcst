"""Common declaration and Tensor-owner contracts; no chart execution methods."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn

from torchcst._coordinate_state import _floating_dtype, _validate_loaded_state
from torchcst._validation import _shape
from torchcst.geometry.spec import GeometrySpec
from torchcst.geometry.state import GeometryState
from torchcst.patterns.spec import (
    GridPatternSpec,
    LinePatternSpec,
    PatternSpec,
    PointsPatternSpec,
)
from torchcst.patterns.state import PatternState


@dataclass(frozen=True, kw_only=True)
class ChartSpec(ABC):
    geometry: GeometrySpec
    shape: tuple[int, ...]
    revision: int = 1

    @property
    @abstractmethod
    def kind(self) -> str:
        """Immutable family identifier supplied by a concrete declaration."""

    @property
    def trainable(self):
        return False

    def __post_init__(self):
        _shape(self.shape)
        if not isinstance(self.geometry, GeometrySpec):
            raise TypeError("chart needs a geometry declaration")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("invalid chart revision")

    @property
    def features(self):
        return math.prod(self.shape)


def _validate_axes(spec):
    if (
        not isinstance(spec.axes, tuple)
        or len(spec.axes) != len(spec.shape)
        or not all(isinstance(a, PatternSpec) for a in spec.axes)
    ):
        raise ValueError("chart needs one pattern per logical axis")
    if any(a.features != n for a, n in zip(spec.axes, spec.shape)):
        raise ValueError("pattern sizes must match chart shape")
    if sum(a.dim for a in spec.axes) != spec.geometry.intrinsic_dim:
        raise ValueError("pattern dimensions differ from geometry")


def _circle_axis(spec):
    offset = 0
    for index, pattern in enumerate(spec.axes):
        if offset == spec.geometry.circle_axis and isinstance(pattern, LinePatternSpec):
            return index
        offset += pattern.dim
    raise ValueError("torus circle_axis must select the chart line axis")


def _validate_torus_span(spec, span):
    if span >= 2 * math.pi * spec.geometry.major_radius:
        raise ValueError("torus chart must span less than one turn")


class ChartState(nn.Module, ABC):
    """Common geometry, logical shape and checkpoint contract of live charts."""

    def __init__(self, spec, *, device=None, dtype=None):
        super().__init__()
        if type(spec) is not self.spec_type or spec.revision != 1:
            raise ValueError("unsupported chart declaration or revision")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.geometry = GeometryState(spec.geometry, device=device, dtype=dtype)
        self.shape, self.features = spec.shape, spec.features

    @property
    @abstractmethod
    def reference(self):
        """A live coordinate Tensor used for device/dtype metadata."""

    @abstractmethod
    def declaration(self) -> ChartSpec:
        """Snapshot live configuration outside execution and Graph capture."""

    @property
    def trainable(self):
        return self.spec.trainable

    @property
    def device(self):
        return self.reference.device

    @property
    def dtype(self):
        return self.reference.dtype

    @property
    def embedding_dim(self):
        return self.geometry.embedding_dim

    @property
    def intrinsic_dim(self):
        return self.geometry.intrinsic_dim

    @property
    def center_parameter_dim(self):
        return self.geometry.center_parameter_dim

    def get_extra_state(self):
        return {
            "format_version": 2,
            "chart_type": type(self.spec).__name__,
            "revision": self.spec.revision,
            "shape": self.shape,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("chart checkpoint contract differs")

    def extra_repr(self):
        return f"shape={self.shape}, geometry={self.geometry.spec.id!r}, trainable={self.trainable}"


class _AxisChartState(ChartState):
    """Internal shared pattern ownership for Product and Strip only."""

    def __init__(self, spec, *, device=None, dtype=None):
        super().__init__(spec, device=device, dtype=dtype)
        if any(
            type(axis) not in (GridPatternSpec, LinePatternSpec, PointsPatternSpec)
            or axis.revision != 1
            for axis in spec.axes
        ):
            raise ValueError("unsupported chart pattern declaration or revision")
        self.axes = nn.ModuleList(
            PatternState(a, device=device, dtype=dtype) for a in spec.axes
        )

    @property
    def reference(self):
        return self.axes[0].reference

    def _declaration_values(self):
        return {
            "geometry": self.geometry.declaration(),
            "axes": tuple(axis.declaration() for axis in self.axes),
        }
