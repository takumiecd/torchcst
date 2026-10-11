"""Mixed-width fixtures preserve live Polar widths and full product semantics."""

import json

import pytest
import torch

from benchmarks.cuda.linear import periodic_comparison as protocol
from benchmarks.cuda.linear import periodic_profile_product as fixture
from benchmarks.cuda.linear.width_scenarios import (
    SCENARIOS,
    definition,
    initial_widths,
    support_summary,
)


@pytest.mark.parametrize("name", SCENARIOS)
def test_frozen_distribution_and_fixture(name, tmp_path):
    case = protocol.case_definition(2048, None, regular_grid=True, width_scenario=name)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    assert protocol.load_case(path) == case
    case["width_distribution"]["weights"][0] += 1
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="frozen"):
        protocol.load_case(path)
    model = fixture.fixture(64, None, atoms=97, regular_grid=True, width_scenario=name)
    actual = fixture.widths(model)
    expected = initial_widths(name, 97, 41)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert torch.equal(expected, initial_widths(name, 97, 41))
    if len(SCENARIOS[name]["values"]) > 1:
        assert not torch.equal(expected, initial_widths(name, 97, 42))
    # Extra width RNG must not move centres or change amplitude signs.
    legacy = fixture.fixture(64, 3, atoms=97, regular_grid=True)
    assert torch.equal(model.atoms.p[:, 2:], legacy.atoms.p[:, 2:])
    amp, _ = fixture.polar_values(model.kernel, model.atoms.p)
    old_amp, _ = fixture.polar_values(legacy.kernel, legacy.atoms.p)
    torch.testing.assert_close(amp, old_amp, rtol=1e-6, atol=1e-6)
    summary = support_summary(model)
    assert sum(summary["rho_bins"].values()) == 97
    for axis in summary["support_count_axes"]:
        assert axis["zero"] + axis["one"] + axis["multiple"] == 97


@pytest.mark.parametrize("name", SCENARIOS)
def test_complete_cpu_operator_and_all_cotangents(name):
    model = fixture.fixture(
        64, None, atoms=19, regular_grid=True, width_scenario=name
    ).double()
    gen = torch.Generator().manual_seed(2541)
    x = torch.randn(3, 64, dtype=torch.float64, generator=gen, requires_grad=True)
    dy = torch.randn(3, 64, dtype=torch.float64, generator=gen)
    y = model(x)
    gx, gp = torch.autograd.grad(y, (x, model.atoms.p), dy)
    truth = fixture.oracle_vjp(model, x, dy, chunk=7)
    for actual, expected in zip((y, gx, gp), truth, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)


def test_subunit_support_is_not_always_one_hot():
    model = fixture.fixture(
        64, None, atoms=3, regular_grid=True, width_scenario="rho0p75"
    ).double()
    with torch.no_grad():
        model.atoms.p[:, 2:] = torch.tensor([[3.5, 3.5], [3.0, 3.0], [3.1, 3.1]])
    summary = support_summary(model)
    assert summary["support_count_axes"][0]["multiple"] == 1
    assert summary["support_count_axes"][0]["one"] == 2
    u, v, _ = fixture.oracle_factors(model.kernel, model.chart, model.atoms.p)
    for axis, raw in zip(summary["support_count_axes"], (u, v), strict=True):
        assert axis["total"] == int((raw > 0).sum())
    assert definition("mix-balanced")["weights"] == [1, 1, 1]


def test_distribution_manifest_cannot_mutate_registry():
    value = definition("mix-narrow")
    value["weights"][0] = 17
    assert definition("mix-narrow")["weights"] == [8, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize(
    "kind", ["matrix-torch", "factor", "onchip-h", "tile-owner-late-bm16"]
)
def test_cuda_mixed_all_cotangents_and_retained_backward(name, kind):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = fixture.fixture(
        64, None, atoms=97, regular_grid=True, width_scenario=name
    ).cuda()
    protocol.bind_plan(model, kind)
    gen = torch.Generator().manual_seed(2541)
    x = torch.randn(3, 64, generator=gen).cuda().requires_grad_()
    dy = torch.randn(3, 64, generator=gen).cuda()
    y = model(x)
    first = torch.autograd.grad(y, (x, model.atoms.p), dy, retain_graph=True)
    second = torch.autograd.grad(y, (x, model.atoms.p), dy)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for label, actual, truth in zip(
        ("Y", "dX", "all_dP"),
        (y, *first),
        fixture.oracle_vjp(model, x, dy),
        strict=True,
    ):
        error = protocol.finite_error(label, actual, truth)
        assert error["max_abs"] <= 4e-4 and error["relative_l2"] <= 4e-4, (label, error)
