"""Two-tile rev6 declaration, independent native H oracle, and legacy guards."""

import json

import numpy as np
import pytest
import torch

from benchmarks.cuda.linear import two_tile_profile_product as ff
from benchmarks.cuda.linear.periodic_comparison import (
    TWO_TILE_CATALOG,
    TWO_TILE_PRIMARY,
    fourier_fixture_module,
    fourier_primary,
    fourier_revision,
    state_gate,
    validate_fourier_record,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("modes", [16, 32])
def test_exact_two_tile_case_and_legacy_distinction(modes):
    case = ff.case_definition(modes=modes)
    assert ff.load_case(case) == case
    assert case["size"] == 1000 and case["atoms"] == 50000
    assert case["periods"] == [100, 100]
    assert case["tile_width"] == 20 and case["local_period"] == 40
    assert case["tile_count"] * case["sites_per_tile"] == case["size"]
    assert fourier_primary(case) == TWO_TILE_PRIMARY
    assert fourier_fixture_module(case) is ff and fourier_revision(case) == "v6"
    model = ff.fixture(case, atoms=11)
    assert model.chart.geometry.periods.dtype == torch.float64
    assert model.chart.origin.dtype == torch.float64
    assert model.atoms.p.dtype == torch.float32
    from benchmarks.cuda.linear.manifest import REGISTRY, decode_catalog

    entries = decode_catalog(json.loads(TWO_TILE_CATALOG.read_text()))
    assert {entry.id for entry in entries} == {"two-tile-shared", "two-tile-direct"}
    for entry in entries:
        assert REGISTRY.loads_plan(REGISTRY.dumps_plan(entry.plan)) == entry.plan


@pytest.mark.parametrize(
    "field,value",
    [
        ("modes", True),
        ("size", 1000.0),
        ("atoms", 50000.0),
        ("local_period", 60),
        ("tile_width", 10),
    ],
)
def test_case_mutations_rejected(field, value):
    case = ff.case_definition()
    case[field] = value
    with pytest.raises(ValueError):
        ff.load_case(case)


@pytest.mark.parametrize("modes", [16, 32])
@pytest.mark.parametrize("floor", [1e-6, 1000.0])
@pytest.mark.parametrize("shifted_origin", [False, True])
def test_native_sites_all_atom_gradients_match_torch_reference(
    modes, floor, shifted_origin
):
    case = ff.case_definition(modes=modes)
    if shifted_origin:
        # Fixture case remains frozen; geometry shift is an explicit oracle test.
        model = ff.fixture(case, atoms=13, size=800, dtype=torch.float64, floor=floor)
        model.chart.origin.copy_(torch.tensor([0.17, -0.21], dtype=torch.float64))
        with torch.no_grad():
            model.atoms.p[:, 2:] += model.chart.origin
    else:
        model = ff.fixture(case, atoms=13, size=800, dtype=torch.float64, floor=floor)
    generator = torch.Generator().manual_seed(928)
    x = torch.randn(3, 800, generator=generator, dtype=torch.float64)
    dy = torch.randn(3, 800, generator=generator, dtype=torch.float64)
    truth = ff.oracle_vjp(model, x, dy, chunk=4)
    xx = x.clone().requires_grad_()
    yy = model(xx)
    gx, gp = torch.autograd.grad(yy, (xx, model.atoms.p), dy)
    for actual, expected in zip((yy.detach(), gx, gp), truth, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)
    assert truth[2][0, 0].abs() > 0
    assert truth[2][0, 2:].count_nonzero() == 0


@pytest.mark.parametrize("modes", [16, 32])
def test_native_h_table_symmetric_normalized(modes):
    profile = (
        ff.fixture(ff.case_definition(modes=modes), atoms=9)
        .kernel.spec.profiles[0]
        .profile
    )
    h = ff.native_h_table(profile, 40)
    np.testing.assert_allclose(h, h[:, ::-1], atol=1e-15)
    np.testing.assert_allclose(h.sum(-1), 1, atol=1e-15)
    assert h.shape == (6, modes + 1) and np.isfinite(h).all()


def test_invalid_chunk_and_failed_worker_no_timings():
    model = ff.fixture(ff.case_definition(), atoms=9)
    with pytest.raises(ValueError, match="positive integer"):
        ff.oracle_vjp(model, torch.zeros(1, 1000), torch.zeros(1, 1000), chunk=0)
    record = {
        "kind": "two-tile-shared",
        "status": "FAIL",
        "failed_stage": "oracle",
        "error": "gate failed",
        "timing": None,
    }
    validate_fourier_record(record)
    record["timing"] = {"median_ms": 1}
    with pytest.raises(ValueError, match="without timings"):
        validate_fourier_record(record)


def test_same_cotangent_public_optimizer_cpu():
    result = state_gate(device="cpu", steps=3, fourier_case=ff.case_definition())
    assert result["status"] == "PASS" and result["final_clock"] == 4
