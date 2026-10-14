"""Frozen dyadic semantics and independent native-site VJP regression gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.cuda.linear import matern_dyadic_product as md
from benchmarks.cuda.linear import periodic_comparison as pc
from benchmarks.cuda.linear.matern_native_oracle import native_periodic


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", [128, 1024])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
def test_dyadic_native_all_site_vjps(size, floor):
    model = md.fixture(
        md.case_definition(), dtype=torch.float64, atoms=17, size=size, floor=floor
    )
    generator = torch.Generator().manual_seed(113)
    x = torch.randn(
        2, size, dtype=torch.float64, generator=generator, requires_grad=True
    )
    dy = torch.randn(2, size, dtype=torch.float64, generator=generator)
    y = model(x)
    dx, dp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    expected = md.oracle_vjp(model, x, dy, chunk=5)
    for actual, truth in zip((y, dx, dp), expected, strict=True):
        torch.testing.assert_close(actual, truth, atol=3e-10, rtol=3e-10)
    radial = (dp[:, :2] * model.atoms.p[:, :2]).sum(-1)
    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=3e-11, rtol=0)
    user = md.user_widths(model)
    torch.testing.assert_close(
        user, torch.logspace(-1, 1, 17, dtype=torch.float64), atol=2e-8, rtol=2e-7
    )


def test_dyadic_repeated_square_profile_matches_native_image_sum():
    distance = np.array([0, 0.025, 0.1, 0.73, 10, 51.2, 102.375])
    period = 102.4
    images = np.arange(-5, 6)
    delta = distance[:, None] + images * period
    r = np.abs(delta)
    peak_r = np.abs(images * period)
    base, peak_base = np.exp(-r / 10), np.exp(-peak_r / 10)
    # Power each individual image, then sum; powering the image sum is not equivalent.
    for exponent in (1, 2, 4, 8, 16, 32, 64, 128):
        decay = 10 / exponent
        actual = ((1 + r / decay) * base).sum(-1)
        peak = ((1 + peak_r / decay) * peak_base).sum()
        truth, _ = native_periodic(distance, np.sqrt(decay), period)
        np.testing.assert_allclose(actual / peak, truth, atol=3e-14, rtol=3e-14)
        base, peak_base = (
            base.square() if hasattr(base, "square") else base**2,
            peak_base**2,
        )


def test_dyadic_frozen_case_matches_json_and_dispatch():
    case = md.case_definition()
    path = (
        Path(__file__).parents[1]
        / "benchmarks/cuda/linear/cases/matern-dyadic-1024-a52428-b32-k8.json"
    )
    assert json.loads(path.read_text()) == md.load_case(case)
    assert pc.profile_protocol(case) is md
    assert pc.profile_catalog(case) == pc.DYADIC_CATALOG
    assert pc.fourier_primary(case) == pc.DYADIC_PRIMARY
    assert pc.worker_schema(case) == "matern-dyadic-worker.research.v1"
    assert case["atoms"] == int(0.05 * 1024**2)
    assert case["user_decay_length_knots"] == [10 / n for n in case["dyadic_powers"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("atoms", 52429),
        ("batch", 32.0),
        ("knot_count", True),
        ("user_sigma_bounds", [0.01, 100]),
        ("required_workers", ["matern-shared", "dense"]),
    ],
)
def test_dyadic_frozen_mutation_rejected(field, value):
    case = copy.deepcopy(md.case_definition())
    case[field] = value
    with pytest.raises(ValueError, match="frozen"):
        md.load_case(case)


def test_dyadic_aliases_refuse_legacy_math_cohort():
    from benchmarks.cuda.linear import matern_mixture_product as mm

    with pytest.raises(ValueError, match="only"):
        pc.fourier_worker(mm.case_definition(), "matern-stream1")
