from __future__ import annotations

from torchcst._backends.torch.geometry import execution as _geometry
from torchcst._backends.torch.kernels import execution as _kernel

"""Dependency buckets must match full site support, including seam/idle atoms."""
import pytest
import torch
from test_strip_torus_gemm_prototype import _model as seam_model
from test_triton_linear import GPU, _model

from torchcst._backends.cuda.algorithms.strip_torus.fused.executor import forward
from torchcst._backends.cuda.algorithms.strip_torus.fused.host import prepare
from torchcst._backends.torch.operators.strip_torus.preparation import execution_plan
from torchcst._backends.torch.operators.strip_torus.support import (
    station_buckets,
    support_layout,
)
from torchcst._backends.torch.operators.strip_torus.tiled import (
    support_mask,
    tiled_linear,
)


@pytest.mark.parametrize("rows,station_rows", [(1, 1), (2, 1), (19, 5)])
def test_buckets_equal_exact_support_and_preserve_each_atom_once(rows, station_rows):
    torch.manual_seed(71)
    layer = _model(rows=rows, station_rows=station_rows, atoms=97).double()
    chart, p = (layer.chart, layer.atoms.p)
    with torch.no_grad():
        coordinates = torch.randn(97, 3, dtype=p.dtype)
        coordinates[:, 0] = torch.linspace(-100, 100, 97)
        p[:, 2:] = _geometry.encode_centers(
            chart.geometry,
            _geometry.lift_chart_coordinates(chart.geometry, coordinates),
        )
    encoded, _, precision = _kernel.coordinate(
        layer.kernel, "tile_parameters", chart, p
    )
    layout = support_layout(
        execution_plan(layer),
        _geometry.decode_centers(chart.geometry, encoded),
        precision,
        chart.tile_shape[0],
    )
    actual = torch.zeros((chart.tile_count, p.shape[0]), dtype=torch.bool)
    for station in range(chart.tile_count):
        for bucket in station_buckets(station, chart.tile_count):
            actual[
                station,
                layout.order[layout.offsets[bucket] : layout.offsets[bucket + 1]],
            ] = True
    assert torch.equal(actual, support_mask(chart, layer.kernel, p))
    assert torch.equal(layout.order.sort().values, torch.arange(p.shape[0]))
    assert layout.offsets[-1] == p.shape[0]
    x = torch.randn(3, chart.shape[1], dtype=p.dtype, requires_grad=True)
    expected = torch.nn.functional.linear(x, layer.dense_weight())
    result = tiled_linear(chart, layer.kernel, x, p, layout, support_layout=True)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    grad = torch.randn_like(expected)
    for a, b in zip(
        torch.autograd.grad(result, (x, p), grad),
        torch.autograd.grad(expected, (x, p), grad),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
def test_seam_shared_atom_is_stored_once_and_gpu_classification_matches(device):
    layer = seam_model().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        point = layer.atoms.p.new_tensor([[75.5, 0, 0]])
        layer.atoms.p[0, 2:] = _geometry.encode_centers(
            layer.chart.geometry,
            _geometry.lift_chart_coordinates(layer.chart.geometry, point),
        )[0]
    expected = prepare(layer, layer.atoms.p, use_triton=False, support_layout=True)
    actual = prepare(layer, layer.atoms.p, support_layout=True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-06, rtol=1e-06)
    offsets = actual[-1]
    assert offsets[8] > offsets[7]
    if device == "cuda":
        x = torch.randn(33, 4, device=device, requires_grad=True)
        reference = torch.nn.functional.linear(x, layer.dense_weight())
        for bm in (16, 32, 64):
            result = forward(layer, x, layer.atoms.p, batch_tile=bm)
            torch.testing.assert_close(result, reference, atol=3e-05, rtol=3e-05)
            grad = torch.randn_like(result)
            got = torch.autograd.grad(result, (x, layer.atoms.p), grad)
            wanted = torch.autograd.grad(
                reference, (x, layer.atoms.p), grad, retain_graph=True
            )
            for a, b in zip(got, wanted, strict=True):
                torch.testing.assert_close(a, b, atol=0.0001, rtol=0.0001)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
def test_all_idle_atoms_keep_zero_input_and_parameter_gradients(device):
    layer = _model(
        device=device, sigma_min=0.3, backend="tiled" if device == "cpu" else "triton"
    )
    with torch.no_grad():
        layer.atoms.p[:, 0] = 0.8
        layer.atoms.p[:, 1] = 1
        coordinates = layer.atoms.p.new_tensor([[8.5, 0, 0]])
        layer.atoms.p[:, 2:] = _geometry.encode_centers(
            layer.chart.geometry,
            _geometry.lift_chart_coordinates(layer.chart.geometry, coordinates),
        )
    assert not support_mask(layer.chart, layer.kernel, layer.atoms.p).any()
    x = torch.randn(2, 21, device=device, requires_grad=True)
    y = layer(x)
    assert not y.any()
    dx, dp = torch.autograd.grad(y.sum(), (x, layer.atoms.p))
    assert not dx.any() and (not dp.any())
