"""Construct live coordinate state at an explicit configuration boundary.

Prefer pure geometry.presets declarations when the layout is static. Tensor
point tables and random sphere sampling enter through these backend functions.
"""

import torch

from torchcst.geometry import presets
from torchcst.geometry.state import ChartState, GeometryState, PatternState


def _spec(value):
    return (
        value.declaration()
        if isinstance(value, (ChartState, GeometryState, PatternState))
        else value
    )


def euclidean(dim):
    return GeometryState(presets.euclidean(dim))


def sphere(intrinsic_dim, **settings):
    return GeometryState(presets.sphere(intrinsic_dim, **settings))


def torus(intrinsic_dim, **settings):
    return GeometryState(presets.torus(intrinsic_dim, **settings))


def line_pattern(size, **settings):
    return PatternState(presets.line_pattern(size, **settings))


def grid_pattern(shape, **settings):
    return PatternState(presets.grid_pattern(shape, **settings))


def points_pattern(coordinates):
    _validate_coordinates(coordinates)
    return PatternState(
        presets.points_pattern(coordinates.detach().cpu().tolist()),
        device=coordinates.device,
        dtype=coordinates.dtype,
    )


def _validate_coordinates(coordinates):
    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a torch.Tensor")
    if coordinates.ndim != 2 or min(coordinates.shape) < 1:
        raise ValueError("coordinates must have shape [sites, dimensions]")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must have a floating-point dtype")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coordinates must be finite")


def points(coordinates, *, geometry=None, trainable=False):
    _validate_coordinates(coordinates)
    return ChartState(
        presets.points(
            coordinates.detach().cpu().tolist(),
            geometry=_spec(geometry),
            trainable=trainable,
        ),
        device=coordinates.device,
        dtype=coordinates.dtype,
    )


def grid(shape, **settings):
    return ChartState(presets.grid(shape, **settings))


def linspace(size, **settings):
    return ChartState(presets.linspace(size, **settings))


def sphere_chart(features, *, intrinsic_dim, trainable=False, **settings):
    from ..geometry import execution

    geometry = sphere(intrinsic_dim, **settings)
    return points(
        execution.sample_sites(geometry, features),
        geometry=geometry,
        trainable=trainable,
    )


def product(shape, axes, *, geometry=None):
    axes = tuple(axes)
    spec = presets.product(
        shape, tuple(_spec(a) for a in axes), geometry=_spec(geometry)
    )
    reference = axes[0].reference if isinstance(axes[0], PatternState) else None
    return ChartState(
        spec,
        device=None if reference is None else reference.device,
        dtype=None if reference is None else reference.dtype,
    )


def strip(shape, tile_shape, *, axes, axis, tile_pitch, geometry=None):
    axes = tuple(axes)
    spec = presets.strip(
        shape,
        tile_shape,
        axes=tuple(_spec(a) for a in axes),
        axis=axis,
        tile_pitch=tile_pitch,
        geometry=_spec(geometry),
    )
    reference = axes[0].reference if isinstance(axes[0], PatternState) else None
    return ChartState(
        spec,
        device=None if reference is None else reference.device,
        dtype=None if reference is None else reference.dtype,
    )
