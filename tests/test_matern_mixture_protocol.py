"""Frozen declaration and independent actual-site mixture Y/dX/all-dP tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cuda.linear import matern_mixture_product as mm
from benchmarks.cuda.linear.matern_native_oracle import (
    _polar,
    native_mixture,
    native_periodic,
    oracle_vjp,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("knots", [8, 16])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("size", [128, 1024])
def test_independent_native_all_sites_all_atom_vjps(knots, floor, size):
    layer = mm.fixture(
        mm.case_definition(knots=knots),
        dtype=torch.float64,
        atoms=17,
        size=size,
        floor=floor,
    )
    generator = torch.Generator().manual_seed(847)
    x = torch.randn(
        2, size, generator=generator, dtype=torch.float64, requires_grad=True
    )
    dy = torch.randn(2, size, generator=generator, dtype=torch.float64)
    y = layer(x)
    dx, dp = torch.autograd.grad(y, (x, layer.atoms.p), dy)
    expected = oracle_vjp(layer, x, dy, chunk=5)
    for actual, truth in zip((y, dx, dp), expected, strict=True):
        torch.testing.assert_close(actual, truth, rtol=3e-10, atol=3e-10)
    assert expected[2][0, 0].abs() > 0
    assert expected[2][0, 2:].count_nonzero() == 0
    radial = (dp[:, :2] * layer.atoms.p[:, :2]).sum(-1)
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=3e-11, rtol=0)


@pytest.mark.parametrize("sigma", [0.1, 0.3, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("period", [12.8, 102.4])
def test_periodic_closed_form_against_explicit_image_sum(sigma, period):
    displacement = np.array([0.0, 0.025, 0.1, period / 2, period - 0.025])
    actual, dd = native_periodic(displacement, sigma, period)
    # 40 decay lengths on each side gives a separately evaluated reference.
    images = max(2, int(np.ceil(40 * sigma**2 / period)))
    delta = displacement[:, None] + np.arange(-images, images + 1) * period
    r = abs(delta) / sigma**2
    raw = ((1 + r) * np.exp(-r)).sum(-1)
    derivative = (-delta / sigma**4 * np.exp(-r)).sum(-1)
    norm = (
        (1 + abs(np.arange(-images, images + 1) * period) / sigma**2)
        * np.exp(-abs(np.arange(-images, images + 1) * period) / sigma**2)
    ).sum()
    np.testing.assert_allclose(actual, raw / norm, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(dd, derivative / norm, rtol=3e-12, atol=3e-14)


def test_independent_mixture_centre_derivative_finite_difference():
    sigma = np.geomspace(0.1, 10, 17)
    distance = np.linspace(0.027, 102.373, 31)[None, :] + np.arange(17)[:, None] * 0.017
    knots = mm.case_definition()["width_knots"]
    _, derivative = native_mixture(distance, sigma, 102.4, knots)
    h = 1e-7
    plus, _ = native_mixture(distance - h, sigma, 102.4, knots)
    minus, _ = native_mixture(distance + h, sigma, 102.4, knots)
    np.testing.assert_allclose(
        derivative, (plus - minus) / (2 * h), rtol=2e-5, atol=2e-8
    )


@pytest.mark.parametrize("knots", [8, 16])
def test_frozen_case_and_source_json_geometry(knots):
    case = mm.case_definition(knots=knots)
    assert mm.load_case(case) == case
    assert case["atoms"] == int(0.05 * 1024 * 1024) == 52428
    directory = Path(__file__).parents[1] / "benchmarks/cuda/linear/cases"
    assert (
        json.loads(
            (directory / f"matern-mixture-1024-a52428-b32-k{knots}.json").read_text()
        )
        == case
    )
    model = mm.fixture(case, atoms=17)
    assert model.chart.geometry.periods.dtype == torch.float64
    assert model.chart.origin.dtype == torch.float64
    assert model.chart.geometry.periods.tolist() == [102.4, 102.4]
    assert model.chart.spacing.tolist() == [0.1, 0.1]
    assert model.atoms.p.dtype == torch.float32
    model.to("cpu")
    assert model.chart.geometry.periods.dtype == torch.float64


@pytest.mark.parametrize(
    "field,value",
    [
        ("size", 1024.0),
        ("atoms", 52429),
        ("batch", 16),
        ("knot_count", True),
        ("periods", [100.0, 100.0]),
        ("task_sigma_stop_gradient", False),
        ("required_workers", ["matern-shared", "dense"]),
    ],
)
def test_frozen_protocol_mutations_rejected(field, value):
    case = copy.deepcopy(mm.case_definition())
    case[field] = value
    with pytest.raises(ValueError, match="frozen"):
        mm.load_case(case)


def test_positive_integer_oracle_chunk_required():
    layer = mm.fixture(mm.case_definition(), atoms=7, size=128)
    with pytest.raises(ValueError, match="positive integer"):
        oracle_vjp(layer, torch.zeros(1, 128), torch.zeros(1, 128), chunk=0)


def test_mixture_catalog_registry_roundtrip():
    from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog

    path = (
        Path(__file__).parents[1] / "benchmarks/cuda/linear/plans-matern-mixture.json"
    )
    entries = decode_catalog(json.loads(path.read_text()))
    assert {entry.id for entry in entries} == {"matern-shared", "matern-direct"}
    for entry in entries:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_zero_and_subguard_polar_vectors_are_finite(dtype):
    layer = mm.fixture(mm.case_definition(), dtype=dtype, atoms=5, size=128)
    tiny = torch.finfo(dtype).tiny
    root = np.sqrt(tiny)
    polar = np.array(
        [[0, 0], [root / 4, root / 4], [root, 0], [root, root], [0, root / 4]],
        dtype=np.float64,
    )
    with torch.no_grad():
        layer.atoms.p[:, :2] = torch.tensor(polar, dtype=dtype)
        layer.atoms.p[:, 2:] = 0
    p = layer.atoms.p.detach().double().numpy()
    amp, sigma, derivative = _polar(layer.kernel, p, tiny=tiny)
    assert np.isfinite(amp).all()
    assert np.isfinite(sigma).all()
    assert np.isfinite(derivative).all()
    # Inside the guarded disk the denominator is constant, including at zero.
    amax = float(layer.kernel.scalar("amplitude_max"))
    expected = amax / root
    np.testing.assert_allclose(derivative[[0, 1, 4], 0], expected, rtol=3e-15)
    np.testing.assert_array_equal(derivative[[0, 1, 4], 1], 0)
    # Clamp equality retains the radial cancellation of the true unit map.
    np.testing.assert_allclose(derivative[2], 0, rtol=0, atol=0)
    generator = torch.Generator().manual_seed(912)
    x = torch.randn(2, 128, generator=generator, dtype=dtype)
    dy = torch.randn(2, 128, generator=generator, dtype=dtype)
    result = oracle_vjp(layer, x, dy, chunk=2)
    assert all(torch.isfinite(value).all() for value in result)
    # Zero amplitude has no centre cotangents, while the guarded amplitude VJP
    # remains finite. FP32-origin guards must also survive a final FP32 cast.
    assert result[2][0, 2:].count_nonzero() == 0
    if dtype == torch.float32:
        assert torch.isfinite(result[2].float()).all()
    live_width = mm.widths(layer)
    assert live_width.device == layer.atoms.p.device
    assert live_width.dtype == torch.float64
    torch.testing.assert_close(live_width, torch.from_numpy(sigma), rtol=0, atol=0)
