"""Abstract chart contracts, per-family ownership and exact backend registration."""

import copy
import inspect
from dataclasses import fields, replace

import pytest
import torch

from torchcst import (
    ChartSpec,
    ChartState,
    CSTLinear,
    ExplicitChartSpec,
    ExplicitChartState,
    ProductChartSpec,
    ProductChartState,
    StripChartSpec,
    StripChartState,
    TriweightSpec,
    chart_presets,
    compile_chart,
    geometry_presets,
    pattern_presets,
    presets,
)
from torchcst._backends.torch.charts import execution


def axes():
    return (pattern_presets.line(4, spacing=0.2), pattern_presets.line(3, spacing=0.3))


def spec(kind):
    if kind == "explicit":
        return chart_presets.points(((0.0, 0.0), (1.0, 0.0)))
    if kind == "product":
        return chart_presets.product((4, 3), axes())
    return chart_presets.strip((4, 3), (2, 3), axes=axes(), axis=0, tile_pitch=0.8)


def test_common_contracts_are_abstract_and_have_only_common_fields():
    assert inspect.isabstract(ChartSpec) and inspect.isabstract(ChartState)
    assert {f.name for f in fields(ChartSpec)} == {"geometry", "shape", "revision"}
    with pytest.raises(TypeError, match="abstract"):
        ChartSpec(geometry=geometry_presets.euclidean(1), shape=(2,))
    with pytest.raises(TypeError, match="abstract"):
        ChartState(spec("product"))
    assert not hasattr(ChartState, "tile_count")
    assert not hasattr(ChartState, "tile_grid")


@pytest.mark.parametrize(
    "kind,spec_type,state_type",
    [
        ("explicit", ExplicitChartSpec, ExplicitChartState),
        ("product", ProductChartSpec, ProductChartState),
        ("strip", StripChartSpec, StripChartState),
    ],
)
def test_compilation_and_snapshots_preserve_concrete_type(kind, spec_type, state_type):
    declaration = spec(kind)
    state = compile_chart(declaration, dtype=torch.float64)
    assert type(declaration) is spec_type and type(state) is state_type
    assert type(state.declaration()) is spec_type
    assert isinstance(state, ChartState)
    assert state.reference.dtype == torch.float64
    for owner in (ChartState, state_type):
        for method in ("positions", "squared_distance", "center_offsets", "retract"):
            assert not hasattr(owner, method)
    with pytest.raises(ValueError, match="unsupported chart"):
        compile_chart(replace(declaration, revision=2))
    with pytest.raises(ValueError, match="unsupported chart"):
        state_type(replace(declaration, revision=2))


def test_each_family_has_its_own_fields_and_tensors():
    explicit, product, strip = (spec(kind) for kind in ("explicit", "product", "strip"))
    for name in ("axes", "tile_shape", "axis", "tile_pitch"):
        assert not hasattr(explicit, name)
        assert not hasattr(compile_chart(explicit), name)
    for name in ("coordinates", "spacing", "tile_shape", "axis", "tile_pitch"):
        assert not hasattr(product, name)
        assert not hasattr(compile_chart(product), name)
    assert "trainable" not in {f.name for f in fields(ProductChartSpec)}
    assert "trainable" not in {f.name for f in fields(StripChartSpec)}
    assert not hasattr(strip, "coordinates") and not hasattr(strip, "spacing")
    assert "tile_pitch" in dict(compile_chart(strip).named_buffers())
    assert "coordinates" in dict(compile_chart(explicit).named_buffers())
    with pytest.raises(TypeError, match="tile_pitch"):
        ProductChartSpec(
            geometry=product.geometry,
            shape=product.shape,
            axes=product.axes,
            tile_pitch=1.0,
        )
    with pytest.raises(TypeError, match="kind"):
        ProductChartSpec(
            geometry=product.geometry,
            shape=product.shape,
            axes=product.axes,
            kind="strip",
        )


def test_custom_types_cannot_impersonate_registered_declarations_or_states():
    class CustomProduct(ProductChartSpec):
        pass

    class CustomState(ProductChartState):
        pass

    product = spec("product")
    with pytest.raises(ValueError, match="unsupported chart"):
        compile_chart(
            CustomProduct(
                geometry=product.geometry, shape=product.shape, axes=product.axes
            )
        )
    with pytest.raises(ValueError, match="unsupported chart"):
        execution.positions(CustomState(product), torch.tensor([0]))
    state = compile_chart(product)
    state.spec = spec("strip")
    with pytest.raises(ValueError, match="unsupported chart"):
        execution.positions(state, torch.tensor([0]))


def test_checkpoint_rejects_other_family_and_old_tagged_format():
    product, strip = compile_chart(spec("product")), compile_chart(spec("strip"))
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        product.load_state_dict(strip.state_dict())
    checkpoint = copy.deepcopy(product.state_dict())
    checkpoint["_extra_state"] = {"format_version": 1, "kind": "product"}
    with pytest.raises(RuntimeError, match="checkpoint contract"):
        product.load_state_dict(checkpoint)


def test_geometry_and_pattern_packages_do_not_export_chart_types_or_constructors():
    import torchcst.geometry
    import torchcst.patterns

    assert not hasattr(torchcst.geometry, "ChartSpec")
    assert not hasattr(torchcst.geometry, "PatternSpec")
    assert not hasattr(torchcst.patterns, "ChartSpec")
    for name in ("strip", "product", "points", "grid"):
        assert not hasattr(geometry_presets, name)
    assert not hasattr(chart_presets, "line_pattern")


def test_explicit_operator_layout_does_not_need_axis_pattern_wrapping():
    coordinates = tuple((float(i), float(j)) for i in range(2) for j in range(3))
    declaration = ExplicitChartSpec(
        geometry=geometry_presets.euclidean(2),
        shape=(2, 3),
        coordinates=coordinates,
        trainable=True,
    )
    model = CSTLinear(
        chart=declaration,
        atoms=2,
        kernel=presets.radial(presets.fixed_profile(TriweightSpec(), 2.0)),
    )
    x = torch.randn(4, 3, requires_grad=True)
    actual = model(x)
    expected = torch.nn.functional.linear(x, model.dense_weight())
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert actual.shape == (4, 2)
    assert torch.isfinite(model.chart.coordinates.grad).all()
