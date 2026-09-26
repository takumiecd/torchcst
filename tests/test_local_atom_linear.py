"""Local contractions against the canonical dense operator and its gradients."""

import pytest
import torch
import torch.nn.functional as F
from test_triton_linear import GPU, _model

from prototypes.local_atom_linear import forward, reference
from torchcst import Biweight, Triangle, Triweight, WendlandC2


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
@pytest.mark.parametrize("profile", [Biweight, Triweight, Triangle, WendlandC2])
@pytest.mark.parametrize("representation", ["intrinsic", "ambient"])
@pytest.mark.parametrize("sigma_min", [0.3, 3.5])
def test_local_values_and_gradients(device, profile, representation, sigma_min):
    torch.manual_seed(45)
    dtype = torch.float64 if device == "cpu" else torch.float32
    layer = _model(
        profile=profile,
        representation=representation,
        sigma_min=sigma_min,
        device=device,
    ).to(dtype)
    p = layer.atoms.p
    with torch.no_grad():
        p[:, 0].uniform_(-0.7, 0.7)
        p[0, 0] = 0  # amplitude derivative must survive zero forward contribution
        p[1, 0] = 1.4
        point = p.new_tensor([[38.2, 0.0, 0.0]])
        p[2, 2:] = layer.chart.geometry.encode_centers(
            layer.chart.geometry.lift_chart_coordinates(point)
        )[0]
    x = torch.randn(21, 5, device=device, dtype=dtype).T.requires_grad_()
    expected = F.linear(x, layer.dense_weight())
    actual = (reference if device == "cpu" else forward)(layer, x, p)
    grad = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, p), grad)
    got = torch.autograd.grad(actual, (x, p), grad)
    tolerance = 1e-10 if device == "cpu" else 1e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for a, b in zip(got, wanted, strict=True):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


@GPU
@pytest.mark.parametrize("rows,station_rows,batch", [(1, 1, 1), (2, 1, 5), (19, 5, 0)])
def test_small_station_counts_and_empty_batch(rows, station_rows, batch):
    layer = _model(rows=rows, station_rows=station_rows, device="cuda")
    x = torch.randn(batch, 21, device="cuda", requires_grad=True)
    expected = F.linear(x, layer.dense_weight())
    actual = forward(layer, x, layer.atoms.p)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), (x, layer.atoms.p)),
        torch.autograd.grad(expected.sum(), (x, layer.atoms.p)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@GPU
def test_idle_and_capture_observe_updated_atoms():
    layer = _model(device="cuda", sigma_min=0.3)
    x = torch.randn(4, 21, device="cuda", requires_grad=True)
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.8
        layer.atoms.p[:, 1] = 1
        coord = layer.atoms.p.new_tensor([[8.5, 0, 0]])
        layer.atoms.p[:, 2:] = layer.chart.geometry.encode_centers(
            layer.chart.geometry.lift_chart_coordinates(coord)
        )
    y = forward(layer, x, layer.atoms.p)
    assert not y.any()
    for grad in torch.autograd.grad(y.sum(), (x, layer.atoms.p)):
        assert not grad.any()
    with torch.no_grad():
        # Warm on another stream, then capture the full preparation and forward.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(layer, x, layer.atoms.p)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(layer, x, layer.atoms.p)
        layer.atoms.p[:, 2].sub_(1.5)
        graph.replay()
        torch.testing.assert_close(
            output, F.linear(x, layer.dense_weight()), atol=3e-5, rtol=3e-5
        )
