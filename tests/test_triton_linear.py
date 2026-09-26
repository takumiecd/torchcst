"""GPU parity gates for generated weights and their first-order gradients."""

import importlib.util
import math

import pytest
import torch

from torchcst import (
    Biweight,
    CSTLinear,
    CSTParameterAdam,
    DirectAmpWidth,
    GridPattern,
    LinePattern,
    ParameterAdamConfig,
    StripChart,
    TorusGeometry,
    Triangle,
    Triweight,
    WendlandC2,
)
from torchcst.nn._backends._preparation import execution_plan, geometry_factors, prepare
from torchcst.optim import LinearJGAtomGrad

GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="requires NVIDIA CUDA and Triton",
)


def _model(
    profile=Triweight,
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
    chart = StripChart(
        shape=(rows, 21),
        tile_shape=(station_rows, 21),
        axes=(
            LinePattern(rows, spacing=min(1.75, 7 / max(station_rows - 1, 1))),
            GridPattern(cross_shape, spacing=0.2),
        ),
        axis=0,
        tile_pitch=10.0,
        geometry=TorusGeometry(
            1 + len(cross_shape),
            major_radius=max(stations, 2) * 10 / (2 * math.pi),
            minor_radius=1.0,
            representation=representation,
        ),
    )
    kernel = DirectAmpWidth(
        amplitude_max=1.0,
        sigma_min=sigma_min,
        sigma_birth=3.5,
        sigma_max=3.5,
        w_c=0.05,
        profile=profile(sigma_min, normalize_columns=False),
        checkpoint_blocks=False,
    )
    return CSTLinear(
        chart=chart, atoms=atoms, kernel=kernel, backend=backend, device=device
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
    expected = chart.positions(torch.arange(chart.features))
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
@pytest.mark.parametrize("profile", [Biweight, Triweight, WendlandC2, Triangle])
@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
@pytest.mark.parametrize("sigma_min", [0.3, 3.5])
def test_triton_matches_dense_forward_and_gradients(profile, representation, sigma_min):
    torch.manual_seed(31)
    dense = _model(profile, representation, device="cuda", sigma_min=sigma_min)
    fused = _model(
        profile, representation, device="cuda", backend="triton", sigma_min=sigma_min
    )
    with torch.no_grad():
        dense.atoms.p[:, 0].uniform_(-0.7, 0.7)
        dense.atoms.p[:, 1].uniform_(1.0, 4.0)
        # Amplitude clipping and the seam must preserve the canonical gradient.
        dense.atoms.p[1, 0] = 1.4
        point = dense.atoms.p.new_tensor([[37.5, 0.0, 0.0]])
        center = dense.chart.geometry.lift_chart_coordinates(point)
        dense.atoms.p[0, 2:] = dense.chart.geometry.encode_centers(center)[0]
    fused.load_state_dict(dense.state_dict())
    x = torch.randn(21, 33, device="cuda").T.requires_grad_()
    x_fused = x.detach().clone().requires_grad_()
    grad = torch.randn(19, 33, device="cuda").T
    expected, actual = dense(x), fused(x_fused)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(x_fused.grad, x.grad, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        fused.atoms.p.grad, dense.atoms.p.grad, atol=1e-4, rtol=1e-4
    )
    assert torch.equal(
        fused.atoms.p.grad[:, 1], torch.zeros_like(fused.atoms.p.grad[:, 1])
    )


@GPU
@pytest.mark.parametrize(
    "rows,station_rows,atoms",
    [(5, 5, 1), (9, 5, 1), (19, 5, 1), (19, 5, 37), (37, 20, 13)],
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
    actual = model(inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (inputs, model.atoms.p), grad)
    actual_grads = torch.autograd.grad(actual, (inputs, model.atoms.p), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


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
        x.grad, model.dense_weight().sum(0), atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@GPU
def test_triton_training_and_custom_atom_gradient_route():
    torch.manual_seed(4)
    model = _model(device="cuda", backend="triton")
    optimizer = CSTParameterAdam(
        model, cst=ParameterAdamConfig(lr=0.001, decay_steps=None)
    )
    inputs = torch.randn(3, 21, device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(
            model(inputs),
            torch.nn.functional.linear(inputs, model.dense_weight()),
            atol=2e-5,
            rtol=2e-5,
        )
    collector = LinearJGAtomGrad(mode="custom", factored=False)
    model.atoms.set_grad(collector)
    collector.begin()
    model(inputs).sum().backward()
    collector.complete()
    p = model.atoms.p.detach().clone().requires_grad_()
    oracle = torch.nn.functional.linear(
        inputs, model.kernel.weight(model.chart, p)
    ).sum()
    expected = torch.autograd.grad(oracle, p)[0]
    torch.testing.assert_close(collector.snapshot().jg, expected, atol=1e-4, rtol=1e-4)


@GPU
def test_triton_rejects_unsupported_dtype_and_deterministic_atom_backward():
    model = _model(device="cuda", backend="triton").double()
    with pytest.raises(TypeError, match="float32"):
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
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    gradient = torch.randn_like(actual)
    wanted = torch.autograd.grad(expected, (x, model.atoms.p), gradient)
    got = torch.autograd.grad(actual, (x, model.atoms.p), gradient)
    for actual_grad, expected_grad in zip(got, wanted):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)


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
        model.kernel.profile = Biweight(3.5, normalize_columns=False)
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
        model.kernel.sigma_max_input.fill_(100.0)
    else:
        model.kernel.profile.normalize_columns = True
    with pytest.raises(ValueError):
        prepare(model, model.atoms.p)


def test_preparation_can_be_warmed_in_inference_then_used_for_training():
    model = _model()
    with torch.inference_mode():
        prepare(model, model.atoms.p)
    plan = execution_plan(model)
    assert not plan.circle.is_inference() and not plan.section.is_inference()
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
    model = _model(device="cuda", backend="triton", sigma_min=0.3)
    x = torch.randn(7, 21, device="cuda")
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = model(x)
        for _ in range(2):
            x.normal_()
            model.atoms.p[:, 0].add_(0.01)
            model.atoms.p[:, 2].add_(1.0)
            graph.replay()
            expected = torch.nn.functional.linear(x, model.dense_weight())
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
