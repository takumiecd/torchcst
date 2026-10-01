"""Coordinate contracts, Tensor ownership and functional execution boundaries."""

import copy
import io
from dataclasses import replace

import pytest
import torch

import torchcst
from torchcst import (
    ChartState,
    CSTConv2d,
    CSTLinear,
    GeometryState,
    PatternState,
    TriweightSpec,
)
from torchcst import (
    geometry_presets as layout,
)
from torchcst import (
    presets as kernels,
)
from torchcst._backends.torch.charts import execution as charts
from torchcst._backends.torch.geometry import execution as geometry
from torchcst._backends.torch.patterns import execution as patterns


def radial():
    return kernels.radial(kernels.fixed_profile(TriweightSpec(), sigma=2))


def matrix(kind="product", **settings):
    args = dict(
        shape=(5, 3),
        axes=(layout.line_pattern(5, spacing=0.2), layout.line_pattern(3, spacing=0.3)),
        **settings,
    )
    if kind == "strip":
        return layout.strip(tile_shape=(2, 3), axis=0, tile_pitch=0.8, **args)
    return layout.product(**args)


def test_legacy_classes_and_declaration_adapters_are_removed():
    for name in (
        "Chart",
        "ExplicitChart",
        "ProductChart",
        "StripChart",
        "Geometry",
        "EuclideanGeometry",
        "SphereGeometry",
        "TorusGeometry",
        "SitePattern",
        "GridPattern",
        "LinePattern",
        "PointsPattern",
    ):
        assert not hasattr(torchcst, name)
    for cls in (ChartState, GeometryState, PatternState):
        for method in (
            "positions",
            "squared_distance",
            "initialize_centers",
            "retract",
            "transport",
        ):
            assert not hasattr(cls, method)


@pytest.mark.parametrize("kind", ["product", "strip"])
def test_single_chart_declaration_constructs_an_operator(kind):
    layer = CSTLinear(chart=matrix(kind), atoms=2, kernel=radial(), dtype=torch.float64)
    inputs = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    expected = torch.nn.functional.linear(inputs, layer.dense_weight())
    torch.testing.assert_close(layer(inputs), expected)
    layer(inputs).square().sum().backward()
    assert inputs.grad is not None and layer.atoms.p.grad is not None
    assert layer.chart.dtype == torch.float64
    assert layer.declaration().layout.chart.kind == kind


def test_pair_declarations_construct_a_conv_and_linear_with_lazy_axes():
    inp = layout.product((4,), (layout.line_pattern(4, spacing=0.2),))
    out = layout.product((3,), (layout.line_pattern(3, spacing=0.3),))
    kernel = kernels.separable(
        input_profile=kernels.fixed_profile(TriweightSpec(), sigma=2),
        output_profile=kernels.fixed_profile(TriweightSpec(), sigma=2),
    )
    linear = CSTLinear(inp, out, atoms=2, kernel=kernel)
    conv = CSTConv2d(
        inp, out, in_channels=1, out_channels=3, kernel_size=2, atoms=2, kernel=kernel
    )
    x = torch.randn(2, 4)
    torch.testing.assert_close(
        linear(x), torch.nn.functional.linear(x, linear.dense_weight())
    )
    image = torch.randn(2, 1, 4, 4)
    torch.testing.assert_close(
        conv(image), torch.nn.functional.conv2d(image, conv.dense_weight())
    )


def test_trainable_explicit_coordinates_are_owned_and_visible_to_autograd():
    spec = layout.points(((0.0,), (0.4,), (0.9,)), trainable=True)
    frozen = layout.points(((0.0,), (0.6,)))
    kernel = kernels.separable(
        input_profile=kernels.fixed_profile(TriweightSpec(), sigma=2),
        output_profile=kernels.fixed_profile(TriweightSpec(), sigma=2),
    )
    layer = CSTLinear(spec, frozen, atoms=2, kernel=kernel)
    assert isinstance(layer.input_chart.coordinates, torch.nn.Parameter)
    layer(torch.randn(2, 3)).square().sum().backward()
    assert torch.isfinite(layer.input_chart.coordinates.grad).all()
    before = layer.input_chart.declaration()
    with torch.no_grad():
        layer.input_chart.coordinates.add_(0.1)
    assert before != layer.input_chart.declaration()
    torch.testing.assert_close(
        torch.tensor(before.axes[0].coordinates), torch.tensor(spec.axes[0].coordinates)
    )


@pytest.mark.parametrize("kind", ["product", "strip"])
def test_coordinate_checkpoint_is_weights_only_and_restores_live_buffers(kind):
    source = ChartState(matrix(kind), dtype=torch.float64)
    source.axes[0].start.add_(0.25)
    target = ChartState(matrix(kind), dtype=torch.float64)
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    target.load_state_dict(torch.load(stream, weights_only=True))
    assert target.declaration() == source.declaration()
    torch.testing.assert_close(
        charts.positions(target, torch.arange(15)),
        charts.positions(source, torch.arange(15)),
    )


@pytest.mark.parametrize(
    "spec, state_type",
    [
        (layout.euclidean(2), GeometryState),
        (layout.line_pattern(3, spacing=0.2), PatternState),
        (matrix(), ChartState),
    ],
)
def test_unsupported_revisions_reject_at_compilation(spec, state_type):
    with pytest.raises(ValueError, match="unsupported"):
        state_type(replace(spec, revision=2))


@pytest.mark.parametrize("representation", ["ambient", "intrinsic"])
def test_sphere_and_torus_chord_distances_match_independent_ambient_oracle(
    representation,
):
    for spec in (
        layout.sphere(2, radius=2, representation=representation),
        layout.torus(2, major_radius=5, minor_radius=1, representation=representation),
    ):
        state = GeometryState(spec, dtype=torch.float64)
        sites = geometry.lift_chart_coordinates(
            state, torch.tensor([[0.1, 0.2], [0.4, -0.1]], dtype=torch.float64)
        )
        centers = geometry.encode_centers(state, sites[:1]).requires_grad_()
        decoded = geometry.decode_centers(state, centers)
        expected = (sites[:, None] - decoded[None, :]).square().sum(-1)
        actual = geometry.squared_distance(state, sites, centers)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), centers, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), centers)[0],
        )


def test_checkpoint_rejects_invalid_loaded_scalars_and_point_tables():
    chart = ChartState(matrix("strip"))
    checkpoint = copy.deepcopy(chart.state_dict())
    checkpoint["tile_pitch"].fill_(0.01)
    with pytest.raises(RuntimeError, match="invalid coordinate"):
        chart.load_state_dict(checkpoint)
    chart = ChartState(layout.points(((1.0, 0.0, 0.0),), geometry=layout.sphere(2)))
    checkpoint = copy.deepcopy(chart.state_dict())
    checkpoint["coordinates"].zero_()
    with pytest.raises(RuntimeError, match="invalid coordinate"):
        chart.load_state_dict(checkpoint)


def test_huge_product_chart_owns_only_axis_state_and_executes_bounded_selection():
    spec = layout.product(
        (1_000_000, 1_000_000),
        (
            layout.line_pattern(1_000_000, spacing=0.2),
            layout.line_pattern(1_000_000, spacing=0.3),
        ),
    )
    state = ChartState(spec)
    assert sum(value.numel() for value in state.buffers()) == 4
    selected = charts.positions(state, torch.tensor([0, 999_999, 999_999_999_999]))
    assert selected.shape == (3, 2)
    torch.testing.assert_close(
        selected[0], torch.tensor([spec.axes[0].start[0], spec.axes[1].start[0]])
    )


def test_execution_reads_live_tensor_state_without_snapshots(monkeypatch):
    state = ChartState(matrix("strip"))

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration snapshots must stay outside execution")

    for cls in (ChartState, GeometryState, PatternState):
        monkeypatch.setattr(cls, "declaration", forbidden)
    centers = charts.initialize_centers(state, 2, mode="balanced").requires_grad_()
    charts.squared_distance(state, centers, slice(0, 3)).sum().backward()
    assert torch.isfinite(centers.grad).all()
    assert patterns.positions(state.axes[0], torch.arange(3)).shape == (3, 1)
