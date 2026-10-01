"""Geometry/layout declarations preserve semantic choices and state ownership."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
import torch

from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst.geometry import (
    ChartSpec,
    EuclideanGeometrySpec,
    GridPatternSpec,
    LinePatternSpec,
    PointsPatternSpec,
    SphereGeometrySpec,
    TorusGeometrySpec,
)


@pytest.mark.parametrize("representation", ["ambient", "intrinsic"])
def test_sphere_and_torus_declare_chord_metric_and_center_storage(representation):
    sphere = _construction.sphere(
        3, radius=2.0, representation=representation
    ).declaration()
    torus = _construction.torus(
        3,
        major_radius=10.0,
        minor_radius=1.5,
        representation=representation,
        max_arc_step=0.2,
    ).declaration()
    assert sphere.metric == torus.metric == "ambient_chord"
    assert sphere.embedding_dim == torus.embedding_dim == 4
    assert (
        sphere.center_parameter_dim
        == torus.center_parameter_dim
        == (4 if representation == "ambient" else 3)
    )
    assert torus.max_arc_step == 0.2
    json.dumps(asdict(torus))
    with pytest.raises(FrozenInstanceError):
        sphere.radius = 4.0


def test_strip_physical_pitch_is_declared_separately_from_execution_settings():
    chart = _construction.strip(
        shape=(5, 6),
        tile_shape=(2, 6),
        axis=0,
        tile_pitch=0.7,
        axes=(
            _construction.line_pattern(5, spacing=0.2),
            _construction.grid_pattern((2, 3), spacing=0.1),
        ),
    )
    spec = chart.declaration()
    assert spec.kind == "strip"
    assert spec.shape == (5, 6)
    assert spec.tile_shape == (2, 6)
    assert spec.tile_pitch == pytest.approx(0.7)
    assert spec.features == 30
    assert isinstance(spec.axes[0], LinePatternSpec)
    assert spec.axes[1].shape == (2, 3)
    chart.tile_pitch.fill_(0.8)
    assert chart.declaration() != spec
    assert spec.tile_pitch == pytest.approx(0.7)
    assert "window_rows" not in asdict(spec)


def test_explicit_trainable_points_remain_live_while_declaration_is_a_snapshot():
    chart = _construction.points(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), trainable=True)
    spec = chart.declaration()
    assert spec.trainable
    assert isinstance(spec.axes[0], PointsPatternSpec)
    assert spec.axes[0].coordinates[0][0] == pytest.approx(0.1)
    keys = set(chart.state_dict())
    _charts.positions(chart, torch.arange(2)).square().sum().backward()
    assert chart.coordinates.grad is not None
    with torch.no_grad():
        chart.coordinates.add_(0.1)
    assert chart.declaration() != spec
    assert spec.axes[0].coordinates[0][0] == pytest.approx(0.1)
    assert set(chart.state_dict()) == keys


def test_points_pattern_snapshots_are_immutable_and_serializable():
    pattern = _construction.points_pattern(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    spec = pattern.declaration()
    pattern.coordinates.add_(1)
    assert spec.features == 2 and spec.dim == 2
    assert spec.coordinates[0][0] == pytest.approx(0.1)
    json.dumps(asdict(spec))


def test_declared_layout_uses_flattened_axis_order_and_singleton_zero_spacing():
    pattern = _construction.grid_pattern((1, 3), low=(0, 0), high=(0, 1))
    assert pattern.declaration().spacing == (0.0, 0.5)
    chart = _construction.product(
        shape=(3, 2),
        axes=(
            _construction.line_pattern(3, spacing=0.2),
            _construction.line_pattern(2, spacing=0.4),
        ),
    )
    spec = chart.declaration()
    assert [p.features for p in spec.axes] == [3, 2]
    assert spec.geometry == EuclideanGeometrySpec(intrinsic_dim=2)


def test_geometry_snapshot_tracks_matching_checkpoint_load():
    source = _construction.sphere(2, radius=3)
    target = _construction.sphere(2, radius=1)
    before = target.declaration()
    target.load_state_dict(source.state_dict())
    assert target.declaration() == source.declaration()
    assert before.radius == 1


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (EuclideanGeometrySpec, {"intrinsic_dim": True}),
        (SphereGeometrySpec, {"intrinsic_dim": 2, "radius": 0}),
        (SphereGeometrySpec, {"intrinsic_dim": 2, "representation": "unknown"}),
        (TorusGeometrySpec, {"intrinsic_dim": 3, "major_radius": 1, "minor_radius": 2}),
        (
            TorusGeometrySpec,
            {
                "intrinsic_dim": 3,
                "major_radius": 10,
                "minor_radius": 1,
                "circle_axis": 3,
            },
        ),
    ],
)
def test_invalid_geometry_declarations_reject(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_layout_validation_rejects_incompatible_geometry_and_tiles():
    line = LinePatternSpec(shape=(3,), start=(0,), spacing=(0.2,))
    with pytest.raises(ValueError, match="dimensions"):
        ChartSpec(
            kind="product",
            geometry=EuclideanGeometrySpec(intrinsic_dim=2),
            shape=(3,),
            axes=(line,),
        )
    with pytest.raises(ValueError, match="tile metadata"):
        ChartSpec(
            kind="product",
            geometry=EuclideanGeometrySpec(intrinsic_dim=1),
            shape=(3,),
            axes=(line,),
            tile_pitch=1,
        )
    with pytest.raises(ValueError, match="disjoint"):
        ChartSpec(
            kind="strip",
            geometry=EuclideanGeometrySpec(intrinsic_dim=1),
            shape=(3,),
            axes=(line,),
            tile_shape=(2,),
            axis=0,
            tile_pitch=0.1,
        )
    grid = GridPatternSpec(shape=(3, 2), start=(0, 0), spacing=(0.2, 0.2))
    with pytest.raises(ValueError, match="line axis"):
        ChartSpec(
            kind="product",
            geometry=TorusGeometrySpec(
                intrinsic_dim=2, major_radius=10, minor_radius=1
            ),
            shape=(6,),
            axes=(grid,),
        )


def test_custom_geometry_and_chart_do_not_inherit_builtin_semantics():
    from torchcst.geometry.state import ChartState, GeometryState

    class CustomGeometry(EuclideanGeometrySpec):
        pass

    class CustomChart(ChartSpec):
        pass

    with pytest.raises(ValueError, match="unsupported geometry"):
        GeometryState(CustomGeometry(intrinsic_dim=2))
    with pytest.raises(ValueError, match="unsupported chart"):
        ChartState(
            CustomChart(
                kind="product",
                shape=(3,),
                axes=(LinePatternSpec(shape=(3,), start=(0,), spacing=(0.2,)),),
                geometry=EuclideanGeometrySpec(intrinsic_dim=1),
            )
        )


def test_config_snapshots_do_not_import_torch_evaluators():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", str(root / "src"))}
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from torchcst import geometry_presets as layout, ChartState, GeometryState
geo = GeometryState(layout.torus(3, major_radius=10, minor_radius=1))
geo.declaration()
chart = ChartState(layout.product(shape=(3,2), axes=(layout.line_pattern(3,spacing=0.2),layout.line_pattern(2,spacing=0.4))))
chart.declaration()
assert not any(name.startswith('torchcst._backends.torch.') for name in sys.modules)
assert 'triton' not in sys.modules
""",
        ],
        env=env,
        check=True,
    )


def test_geometry_execution_does_not_use_configuration_snapshots(monkeypatch):
    geometry = _construction.sphere(2, representation="intrinsic")

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshots must stay outside execution")

    monkeypatch.setattr(type(geometry), "declaration", forbidden)
    centers = torch.tensor([[0.1, 0.2], [0.2, 0.3]], requires_grad=True)
    sites = _geometry.decode_centers(geometry, centers)
    _geometry.squared_distance(geometry, sites, centers).sum().backward()
    assert torch.isfinite(centers.grad).all()
