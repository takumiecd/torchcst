"""Tensor ownership for coordinate declarations; no coordinate evaluators."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import torch
from torch import nn

from .spec import (
    ChartSpec,
    EuclideanGeometrySpec,
    GeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
    PatternSpec,
    PointsPatternSpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


def _coordinates(tensor):
    return tuple(tuple(row) for row in tensor.detach().cpu().tolist())


def _floating_dtype(dtype):
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("coordinate state requires a floating-point dtype")
    return dtype


def _validate_loaded_state(state, incompatible_keys):
    # Loading is an explicit boundary; validate current buffers rather than the
    # constructor snapshot. These reads never occur in the training hot path.
    try:
        state.declaration()
        if isinstance(state, ChartState) and state.spec.kind == "explicit":
            from torchcst._backends.torch.geometry.execution import validate_points

            validate_points(state.geometry, state.coordinates, name="coordinates")
    except (ValueError, TypeError) as error:
        raise RuntimeError("invalid coordinate checkpoint configuration") from error


class GeometryState(nn.Module):
    """Own fixed scalar tensors, while the immutable spec defines their meaning."""

    def __init__(self, spec: GeometrySpec, *, device=None, dtype=None):
        super().__init__()
        if (
            type(spec)
            not in (EuclideanGeometrySpec, SphereGeometrySpec, TorusGeometrySpec)
            or spec.revision != 1
        ):
            raise ValueError("unsupported geometry declaration or revision")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.intrinsic_dim = spec.intrinsic_dim
        self.embedding_dim = spec.embedding_dim
        self.center_parameter_dim = spec.center_parameter_dim
        self.representation = getattr(spec, "representation", "ambient")
        self.circle_axis = getattr(spec, "circle_axis", None)
        self.max_arc_step = getattr(spec, "max_arc_step", None)
        for name in ("radius", "major_radius", "minor_radius", "chart_margin"):
            if hasattr(spec, name):
                self.register_buffer(
                    name,
                    torch.tensor(
                        getattr(spec, name),
                        device=device,
                        dtype=dtype or torch.get_default_dtype(),
                    ),
                )

    def declaration(self) -> GeometrySpec:
        """Snapshot current scalar buffers only at a configuration boundary."""
        return replace(
            self.spec,
            **{
                name: float(value.detach())
                for name, value in self.named_buffers(recurse=False)
            },
        )

    def get_extra_state(self):
        values = asdict(self.spec)
        for name in self._buffers:
            values.pop(name)
        return {"format_version": 1, "geometry": values}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("geometry checkpoint contract differs")


class PatternState(nn.Module):
    """Own one small site pattern, never a Cartesian product table."""

    def __init__(self, spec: PatternSpec, *, device=None, dtype=None):
        super().__init__()
        if (
            type(spec) not in (GridPatternSpec, LinePatternSpec, PointsPatternSpec)
            or spec.revision != 1
        ):
            raise ValueError("unsupported pattern declaration")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.features, self.dim = spec.features, spec.dim
        dtype = dtype or torch.get_default_dtype()
        if isinstance(spec, GridPatternSpec):
            self.shape = spec.shape
            self.register_buffer(
                "start", torch.tensor(spec.start, device=device, dtype=dtype)
            )
            self.register_buffer(
                "spacing", torch.tensor(spec.spacing, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                "coordinates",
                torch.tensor(spec.coordinates, device=device, dtype=dtype),
            )

    @property
    def reference(self):
        return (
            self.start if isinstance(self.spec, GridPatternSpec) else self.coordinates
        )

    def declaration(self) -> PatternSpec:
        if isinstance(self.spec, GridPatternSpec):
            return replace(
                self.spec,
                start=tuple(self.start.detach().cpu().tolist()),
                spacing=tuple(self.spacing.detach().cpu().tolist()),
            )
        return replace(self.spec, coordinates=_coordinates(self.coordinates))

    def get_extra_state(self):
        return {
            "format_version": 1,
            "pattern": type(self.spec).__name__,
            "revision": self.spec.revision,
            "features": self.features,
            "dim": self.dim,
            "shape": getattr(self, "shape", None),
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("site pattern checkpoint contract differs")


class ChartState(nn.Module):
    """Own a chart's live tensors; geometry/layout are declared separately.

    Only explicit charts own a full point table. Product and Strip charts own
    axis patterns and generate requested positions in the selected backend.
    """

    def __init__(self, spec: ChartSpec, *, device=None, dtype=None):
        super().__init__()
        if type(spec) is not ChartSpec or spec.revision != 1:
            raise ValueError("unsupported chart declaration or revision")
        if any(
            type(axis) not in (GridPatternSpec, LinePatternSpec, PointsPatternSpec)
            or axis.revision != 1
            for axis in spec.axes
        ):
            raise ValueError("unsupported chart pattern declaration or revision")
        dtype = _floating_dtype(dtype)
        self.register_load_state_dict_post_hook(_validate_loaded_state)
        self.spec = spec
        self.geometry = GeometryState(spec.geometry, device=device, dtype=dtype)
        self.shape, self.features = spec.shape, spec.features
        self.trainable = spec.trainable
        self.axis, self.tile_shape = spec.axis, spec.tile_shape
        dtype = dtype or torch.get_default_dtype()
        if spec.kind == "explicit":
            coordinates = torch.tensor(
                spec.axes[0].coordinates, device=device, dtype=dtype
            )
            from torchcst._backends.torch.geometry.execution import validate_points

            validate_points(self.geometry, coordinates, name="coordinates")
            if self.trainable:
                self.coordinates = nn.Parameter(coordinates)
            else:
                self.register_buffer("coordinates", coordinates)
            self.register_buffer(
                "spacing",
                None
                if spec.spacing is None
                else torch.tensor(spec.spacing, device=device, dtype=dtype),
            )
        else:
            self.axes = nn.ModuleList(
                PatternState(axis, device=device, dtype=dtype) for axis in spec.axes
            )
            self.register_buffer("spacing", None)
            if spec.kind == "strip":
                self.register_buffer(
                    "tile_pitch",
                    torch.tensor(spec.tile_pitch, device=device, dtype=dtype),
                )

    @property
    def reference(self):
        return (
            self.coordinates if self.spec.kind == "explicit" else self.axes[0].reference
        )

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

    @property
    def tile_grid(self):
        if self.spec.kind != "strip":
            raise TypeError("tile_grid requires a Strip declaration")
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.tile_shape))

    @property
    def tile_count(self):
        return math.prod(self.tile_grid)

    def declaration(self) -> ChartSpec:
        """Snapshot live coordinates/configuration outside execution and capture."""
        values = {"geometry": self.geometry.declaration()}
        if self.spec.kind == "explicit":
            values["axes"] = (
                PointsPatternSpec(coordinates=_coordinates(self.coordinates)),
            )
            values["spacing"] = (
                None
                if self.spacing is None
                else tuple(self.spacing.detach().cpu().tolist())
            )
        else:
            values["axes"] = tuple(axis.declaration() for axis in self.axes)
            if self.spec.kind == "strip":
                values["tile_pitch"] = float(self.tile_pitch.detach())
        return replace(self.spec, **values)

    def get_extra_state(self):
        return {
            "format_version": 1,
            "kind": self.spec.kind,
            "revision": self.spec.revision,
            "shape": self.shape,
            "trainable": self.trainable,
            "tile_shape": self.tile_shape,
            "axis": self.axis,
            "spacing": self.spacing is not None,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("chart checkpoint contract differs")

    def extra_repr(self):
        return f"kind={self.spec.kind!r}, shape={self.shape}, geometry={self.geometry.spec.id!r}, trainable={self.trainable}"
