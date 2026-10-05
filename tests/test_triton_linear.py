from __future__ import annotations

from torchcst import BandwidthBounds, BiweightSpec, CSTOptimizer, TriweightSpec, presets
from torchcst._backends.torch.charts import construction as _construction
from torchcst._backends.torch.charts import execution as _charts
from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel
from torchcst._backends.torch.kernels.execution import KernelOptions
from torchcst.kernels.state import ProfileState

"GPU parity gates for generated weights and their first-order gradients."
import importlib.util
import math

import pytest
import torch
from kernel_cases import (
    biweight_state,
    direct_state,
    triangle_state,
    triweight_state,
    wendland_state,
)

from torchcst import CSTLinear
from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.executor import (
    forward as triton_forward,
)
from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.host import prepare
from torchcst._backends.dispatch import UnsupportedPlan
from torchcst._backends.torch.operators.strip_torus.preparation import (
    execution_plan,
    geometry_factors,
)

GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="requires NVIDIA CUDA and Triton",
)


def _model(
    profile=triweight_state,
    representation="intrinsic",
    *,
    atoms=13,
    rows=19,
    station_rows=5,
    cross_shape=(3, 7),
    sigma_min=3.5,
    device="cpu",
    backend="materialized",
):
    stations = math.ceil(rows / station_rows)
    chart = _construction.strip(
        shape=(rows, 21),
        tile_shape=(station_rows, 21),
        axes=(
            _construction.line_pattern(
                rows, spacing=min(1.75, 7 / max(station_rows - 1, 1))
            ),
            _construction.grid_pattern(cross_shape, spacing=0.2),
        ),
        axis=0,
        tile_pitch=10.0,
        geometry=_construction.torus(
            1 + len(cross_shape),
            major_radius=max(stations, 2) * 10 / (2 * math.pi),
            minor_radius=1.0,
            representation=representation,
        ),
    )
    kernel = direct_state(
        amplitude_max=1.0,
        w_c=0.05,
        profile=profile(sigma_min, normalize_columns=False),
        input_bounds=BandwidthBounds(
            minimum=sigma_min, maximum=3.5, birth=3.5, upper_floor=sigma_min
        ),
        options=KernelOptions(checkpoint_blocks=False),
        composition="radial",
    )
    return CSTLinear(
        chart=chart,
        atoms=atoms,
        kernel=kernel.declaration(),
        backend=backend,
        device=device,
    )


def test_geometry_factors_reproduce_chart_sites():
    model = _model().double()
    chart = model.chart
    circle, section = geometry_factors(chart)
    actual = torch.cat(
        (
            circle[:, None, :] * section[None, :, :1],
            section[None, :, 1:].expand(chart.shape[0], -1, -1),
        ),
        dim=-1,
    ).reshape(-1, chart.embedding_dim)
    expected = _charts.positions(chart, torch.arange(chart.features))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_triton_backend_rejects_cpu_at_execution():
    model = _model(backend="triton")
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        model(torch.randn(2, 21))


def test_backend_switch_is_validated():
    model = _model()
    with pytest.raises(ValueError, match="factorized"):
        model.backend = "factored"
    with pytest.raises(ValueError, match="backend must be"):
        model.backend = "unknown"


@GPU
@pytest.mark.parametrize(
    "profile", [biweight_state, triweight_state, wendland_state, triangle_state]
)
@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
@pytest.mark.parametrize("sigma_min", [0.3, 3.5])
@pytest.mark.parametrize("atoms", [13, 67])
def test_triton_matches_dense_forward_and_gradients(
    profile, representation, sigma_min, atoms
):
    torch.manual_seed(31)
    dense = _model(
        profile, representation, device="cuda", sigma_min=sigma_min, atoms=atoms
    )
    fused = _model(
        profile,
        representation,
        device="cuda",
        backend="triton",
        sigma_min=sigma_min,
        atoms=atoms,
    )
    with torch.no_grad():
        dense.atoms.p[:, 0].uniform_(-0.7, 0.7)
        dense.atoms.p[:, 1].uniform_(1.0, 4.0)
        dense.atoms.p[1, 0] = 1.4
        point = dense.atoms.p.new_tensor([[37.5, 0.0, 0.0]])
        center = _geometry.lift_chart_coordinates(dense.chart.geometry, point)
        dense.atoms.p[0, 2:] = _geometry.encode_centers(dense.chart.geometry, center)[0]
    fused.load_state_dict(dense.state_dict())
    x = torch.randn(21, 33, device="cuda").T.requires_grad_()
    x_fused = x.detach().clone().requires_grad_()
    grad = torch.randn(19, 33, device="cuda").T
    expected = dense(x)
    actual = triton_forward(fused, x_fused, fused.atoms.p, split_reductions=atoms == 67)
    torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(x_fused.grad, x.grad, atol=3e-05, rtol=3e-05)
    torch.testing.assert_close(
        fused.atoms.p.grad, dense.atoms.p.grad, atol=0.0001, rtol=0.0001
    )
    assert torch.equal(
        fused.atoms.p.grad[:, 1], torch.zeros_like(fused.atoms.p.grad[:, 1])
    )


@GPU
@pytest.mark.parametrize(
    "rows,station_rows,atoms",
    [
        (5, 5, 1),
        (9, 5, 1),
        (19, 5, 1),
        (19, 5, 37),
        (37, 20, 13),
        (37, 20, 67),
        (21, 5, 67),
    ],
)
def test_triton_small_station_counts_and_empty_blocks(rows, station_rows, atoms):
    torch.manual_seed(9)
    model = _model(
        rows=rows,
        station_rows=station_rows,
        atoms=atoms,
        device="cuda",
        backend="triton",
    )
    inputs = torch.randn(2, 3, 21, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(inputs, model.dense_weight())
    actual = triton_forward(model, inputs, model.atoms.p, split_reductions=True)
    torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (inputs, model.atoms.p), grad)
    actual_grads = torch.autograd.grad(actual, (inputs, model.atoms.p), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.0001, rtol=0.0001)


@GPU
def test_triton_empty_batch_and_input_only_gradient():
    model = _model(device="cuda", backend="triton")
    empty = torch.empty(0, 21, device="cuda", requires_grad=True)
    model(empty).sum().backward()
    assert empty.grad.shape == empty.shape
    assert not bool(model.atoms.p.grad.any())
    model.atoms.p.requires_grad_(False)
    x = torch.randn(21, device="cuda", requires_grad=True)
    actual = model(x)
    expected = torch.nn.functional.linear(x, model.dense_weight())
    actual.sum().backward()
    torch.testing.assert_close(
        x.grad, model.dense_weight().sum(0), atol=2e-05, rtol=2e-05
    )
    torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)


@GPU
def test_triton_rejects_unsupported_dtype_and_deterministic_atom_backward():
    model = _model(device="cuda", backend="triton").double()
    with pytest.raises(UnsupportedPlan, match="float32"):
        model(torch.randn(1, 21, device="cuda", dtype=torch.float64))
    model = model.float()
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="atomic accumulation"):
            model(torch.randn(1, 21, device="cuda")).sum().backward()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@GPU
@pytest.mark.parametrize("cross_shape", [(21,), (1, 3, 7)])
def test_triton_torus_embedding_dimensions(cross_shape):
    torch.manual_seed(11)
    model = _model(device="cuda", backend="triton", cross_shape=cross_shape)
    x = torch.randn(17, 21, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(x, model.dense_weight())
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)
    gradient = torch.randn_like(actual)
    wanted = torch.autograd.grad(expected, (x, model.atoms.p), gradient)
    got = torch.autograd.grad(actual, (x, model.atoms.p), gradient)
    for actual_grad, expected_grad in zip(got, wanted):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.0001, rtol=0.0001)


def test_preparation_reuses_only_constants_and_keeps_fresh_autograd_graphs():
    model = _model()
    plan = execution_plan(model)
    state_keys = tuple(model.state_dict())
    first = prepare(model, model.atoms.p)
    first[0].sum().backward()
    assert bool(torch.isfinite(model.atoms.p.grad).all())
    model.atoms.p.grad = None
    with torch.no_grad():
        model.atoms.p[:, 0].add_(0.01)
    second = prepare(model, model.atoms.p)
    second[0].sum().backward()
    assert bool(torch.isfinite(model.atoms.p.grad).all())
    assert execution_plan(model) is plan
    assert first[1] is second[1] and first[2] is second[2]
    assert first[0] is not second[0]
    assert not torch.equal(first[0], second[0])
    assert tuple(model.state_dict()) == state_keys


@pytest.mark.parametrize(
    "change", ["buffer_update", "buffer_replace", "checkpoint", "dtype", "profile"]
)
def test_preparation_invalidates_changed_configuration(change):
    model = _model()
    first = execution_plan(model)
    if change == "buffer_update":
        model.chart.axes[1].start.add_(0.01)
    elif change == "buffer_replace":
        model.chart.tile_pitch = model.chart.tile_pitch.clone()
    elif change == "checkpoint":
        model.load_state_dict(model.state_dict())
    elif change == "dtype":
        model.double()
    else:
        model.kernel.profiles[0] = ProfileState(
            presets.profile(BiweightSpec(), normalize=False)
        )
    second = execution_plan(model)
    assert second is not first
    circle, section = geometry_factors(model.chart)
    torch.testing.assert_close(second.circle, circle)
    torch.testing.assert_close(second.section, section)
    assert execution_plan(model) is second


@pytest.mark.parametrize("change", ["bandwidth", "normalization"])
def test_preparation_revalidates_invalid_configuration(change):
    model = _model()
    execution_plan(model)
    if change == "bandwidth":
        model.kernel.scalar("sigma_max_input").fill_(100.0)
    else:
        model.kernel.profiles[0] = ProfileState(presets.profile(TriweightSpec()))
    with pytest.raises(ValueError):
        prepare(model, model.atoms.p)


def test_preparation_can_be_warmed_in_inference_then_used_for_training():
    model = _model()
    with torch.inference_mode():
        prepare(model, model.atoms.p)
    plan = execution_plan(model)
    assert not plan.circle.is_inference() and (not plan.section.is_inference())
    prepare(model, model.atoms.p)[0].sum().backward()
    assert bool(torch.isfinite(model.atoms.p.grad).all())
    with torch.inference_mode():
        inference_model = _model()
        first = prepare(inference_model, inference_model.atoms.p)
        second = prepare(inference_model, inference_model.atoms.p)
        torch.testing.assert_close(first[0], second[0])
        assert inference_model._strip_torus_plan is None


def test_warm_preparation_does_not_read_tensor_values_on_the_host():
    model = _model()
    prepare(model, model.atoms.p)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as trace:
        prepare(model, model.atoms.p)
    names = {event.key for event in trace.key_averages()}
    assert "aten::item" not in names
    assert "aten::_local_scalar_dense" not in names


@GPU
def test_triton_warm_forward_captures_updated_inputs_and_atoms():
    torch.manual_seed(74)
    model = _model(device="cuda", backend="triton", sigma_min=0.3, atoms=67)
    x = torch.randn(7, 21, device="cuda")

    def run():
        return triton_forward(model, x, model.atoms.p, split_reductions=True)

    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for _ in range(2):
            x.normal_()
            model.atoms.p[:, 0].add_(0.01)
            model.atoms.p[:, 2].add_(1.0)
            graph.replay()
            expected = torch.nn.functional.linear(x, model.dense_weight())
            torch.testing.assert_close(actual, expected, atol=2e-05, rtol=2e-05)


@GPU
@pytest.mark.parametrize("route", ["local", "exhaustive", "chunked"])
def test_routing_boundary_and_neighbors_match_reference_on_graph_replay(route):
    from dataclasses import replace

    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        route_and_layout,
    )
    from torchcst._backends.torch.operators.strip_torus.layout import _station_layout

    model = _model(rows=64, station_rows=4, device="cuda")
    routing = execution_plan(model).routing
    if route == "exhaustive":
        routing = replace(routing, local_candidates=False)
    elif route == "chunked":
        # Repeated intervals exercise equal-distance ties across scan chunks.
        routing = replace(
            routing,
            starts=routing.starts.repeat(65),
            spans=routing.spans.repeat(65),
            last_row=routing.last_row.repeat(65),
            local_candidates=False,
        )
    boundary = torch.tensor([-0.7730104923248291, 0.6343932747840881], device="cuda")
    neighbors = torch.stack(
        (
            boundary,
            torch.nextafter(boundary, torch.full_like(boundary, -torch.inf)),
            torch.nextafter(boundary, torch.full_like(boundary, torch.inf)),
        )
    )
    # Preserve noncontiguous input coverage after moving angle evaluation.
    storage = torch.empty((3, 4), device="cuda")
    decoded = storage[:, ::2]
    decoded.copy_(neighbors)

    def check(actual):
        owners, order, offsets = actual
        expected = routing.owners(decoded)
        layout = _station_layout(expected, routing.starts.numel())
        assert torch.equal(owners, expected)
        assert torch.equal(order, layout.order)
        assert torch.equal(offsets, layout.offsets)

    check(route_and_layout(routing, decoded))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            route_and_layout(routing, decoded)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = route_and_layout(routing, decoded)
    for centers in (neighbors, -neighbors, neighbors.flip(0)):
        decoded.copy_(centers)
        graph.replay()
        check(actual)


@GPU
@pytest.mark.parametrize("rows,station_rows", [(1, 1), (9, 5), (21, 5), (64, 4)])
@pytest.mark.parametrize("count", [0, 37, 257])
def test_fused_routing_matches_reference_including_seams(rows, station_rows, count):
    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        route_and_layout,
    )
    from torchcst._backends.torch.operators.strip_torus.layout import _station_layout

    torch.manual_seed(91)
    model = _model(rows=rows, station_rows=station_rows, device="cuda")
    routing = execution_plan(model).routing
    arcs = torch.linspace(-20, 100, count, device="cuda")
    angle = arcs / routing.major_radius
    decoded = torch.stack((angle.cos(), angle.sin()), dim=1)
    if count:
        decoded[:4] = decoded.new_tensor([[1, 0], [1, -0.0], [-1, 0], [-1, -0.0]])
    owners, order, offsets = route_and_layout(routing, decoded)
    expected = routing.owners(decoded)
    layout = _station_layout(expected, model.chart.tile_count)
    assert torch.equal(owners, expected)
    assert torch.equal(order, layout.order)
    assert torch.equal(offsets, layout.offsets)


@GPU
@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
def test_fused_preparation_matches_values_and_gradients(representation):
    torch.manual_seed(93)
    model = _model(
        representation=representation, atoms=67, device="cuda", sigma_min=0.3
    )
    p = model.atoms.p
    reference = prepare(model, p, use_triton=False)
    actual = prepare(model, p)
    torch.testing.assert_close(actual[0][:, 0], reference[0][:, 0], atol=0, rtol=0)
    torch.testing.assert_close(actual[0][:, 2:], reference[0][:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(
        actual[0][:, 1], reference[0][:, 1], atol=1e-06, rtol=1e-06
    )
    assert torch.equal(actual[3], reference[3])
    gradient = torch.randn_like(actual[0].T).T
    expected = torch.autograd.grad(reference[0], p, gradient)[0]
    got = torch.autograd.grad(actual[0], p, gradient)[0]
    torch.testing.assert_close(got, expected, atol=1e-06, rtol=1e-06)


@GPU
@pytest.mark.parametrize("power,floor,birth", [(1.0, 0.3, 3.5), (0.4, 1.1, 2.2)])
def test_fused_bandwidth_preserves_envelopes_and_stop_gradient(power, floor, birth):
    from torchcst._backends.cuda.algorithms.linear.strip_torus.fused.preparation import (
        tile_parameters,
    )

    torch.manual_seed(94)
    model = _model(atoms=259, device="cuda", sigma_min=0.3)
    kernel = model.kernel
    with torch.no_grad():
        kernel.scalar("upper_decay_power").fill_(power)
        kernel.scalar("upper_floor_input").fill_(floor)
        kernel.scalar("sigma_birth_input").fill_(birth)
        kernel.scalar("lower_kappa").fill_(5.0)
    p = torch.randn(7, 259, device="cuda").T.requires_grad_()
    with torch.no_grad():
        p[:, 0].uniform_(-2, 2)
        p[:, 1].uniform_(-2, 7)
        p[:5, 0] = p.new_tensor([-1, 1, 0, -2, 2])
        p[:5, 1] = p.new_tensor([1, 4, 1, 4, 2.5])
    _, amplitude, precision = tile_parameters(kernel, p)
    _, expected_amplitude, expected_precision = _kernel.coordinate(
        kernel, "_tile_parameters", p
    )
    torch.testing.assert_close(amplitude, expected_amplitude, atol=0, rtol=0)
    torch.testing.assert_close(precision, expected_precision, atol=2e-06, rtol=1e-06)
    assert not precision.requires_grad
    actual_grad = torch.autograd.grad(amplitude.sum(), p)[0]
    expected_grad = torch.autograd.grad(expected_amplitude.sum(), p)[0]
    assert torch.equal(actual_grad, expected_grad)


@GPU
def test_triton_training_with_cst_optimizer():
    torch.manual_seed(4)
    model = _model(device="cuda", backend="triton")
    optimizer = CSTOptimizer(
        torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0), model=model
    )
    inputs = torch.randn(3, 21, device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(
            model(inputs),
            torch.nn.functional.linear(inputs, model.dense_weight()),
            atol=2e-05,
            rtol=2e-05,
        )
