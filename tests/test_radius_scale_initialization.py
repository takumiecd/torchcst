"""Bounded scale export agrees with public weights and all task gradients."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear.radius_cell_mass import fixture
from benchmarks.training import cell_mass_mnist as mnist
from benchmarks.training.owner_window_fp32 import quadrature
from benchmarks.training.radius_local_reference import (
    LOCAL_EXPORT_POLICY,
    bounded_local_factors,
    local_dense_weight,
    local_reference,
)


def site(width=128, count=19):
    return fixture(SimpleNamespace(size=width, atoms=count, rows=4, seed=41))


@pytest.mark.parametrize("wide", [False, True])
def test_local_dense_export_matches_public_weight_and_is_chunk_invariant(wide):
    model, _, _, _ = site()
    with torch.no_grad():
        model.atoms.p[:, 2] = torch.linspace(-0.5, 256.5, model.atoms.count)
        model.atoms.p[:, 3] = torch.linspace(127.4999, -128.5001, model.atoms.count)
        if wide:
            model.atoms.p[:, :2] = torch.tensor([0.0, 2.0])
            model.atoms.p[:, 0] = torch.linspace(-0.01, 0.01, model.atoms.count)
        expected = model.dense_weight()
    actual = local_dense_weight(
        model, chunk_atoms=5, maximum_outer_product_values=65536
    )
    again = local_dense_weight(
        model, chunk_atoms=11, maximum_outer_product_values=131072
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(actual.view(torch.uint8), again.view(torch.uint8))
    factors = bounded_local_factors(model)
    assert factors.input_sites.shape[1] <= model.in_features
    assert factors.output_sites.shape[1] <= model.out_features


@pytest.mark.parametrize("width", [128, 2048])
def test_local_response_and_all_task_gradients_match_public_reference(width):
    model, x, _, dy = site(width=width, count=11)
    model = model.double()
    with torch.no_grad():
        model.atoms.p[:, 2] = torch.linspace(-0.5001, 2 * width + 0.5001, 11)
        model.atoms.p[:, 3] = width - 0.5001
    actual_x = x.double().requires_grad_()
    actual = local_reference(model, actual_x)
    actual_grad = torch.autograd.grad(actual, (actual_x, model.atoms.p), dy.double())
    reference_x = x.double().requires_grad_()
    reference = model(reference_x)
    reference_grad = torch.autograd.grad(
        reference, (reference_x, model.atoms.p), dy.double()
    )
    for left, right in zip((actual, *actual_grad), (reference, *reference_grad)):
        torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-8)


def test_scale_quadrature_uses_actual_chart_width():
    model, x, _, _ = site(width=2048, count=5)
    model = model.double()
    with torch.no_grad():
        model.atoms.p[:, 2] = torch.tensor([1025.25, 2047.5, -0.5, 4096.25, 1536.75])
        model.atoms.p[:, 3] = torch.tensor([2047.5, 1025.25, 4096.25, -0.5, 1800.75])
    got = quadrature(model, x.double(), model.atoms.p)
    torch.testing.assert_close(got, model(x.double()), atol=2e-10, rtol=2e-8)


@pytest.mark.parametrize("width", mnist.RADIUS_SCALE_WIDTHS)
def test_scale_widths_are_distinct_explicit_cohorts(width):
    assert str(width) in mnist.radius_cohort(width)
    assert mnist.radius_cohort(width) not in (
        mnist.RADIUS_COHORT,
        mnist.RADIUS_LARGE_COHORT,
    )
    with pytest.raises(ValueError, match="explicit global-response4"):
        mnist._validate_radius_execution_width(width, "cached", 2, None)
    mnist._validate_radius_execution_width(width, "global-response4", 2, None)


def test_large_initializer_uses_bounded_export_and_validates_policy(monkeypatch):
    called = []
    # Cohort wiring/policy without repeating the full exporter. Its numerical
    # output and VJP are independently exercised above.
    from benchmarks.training import radius_local_reference as local

    def export(model):
        called.append((model.in_features, model.atoms.count))
        return torch.zeros((model.out_features, model.in_features))

    monkeypatch.setattr(local, "local_dense_weight", export)
    initial = mnist.make_radius_initialization(2048)
    assert called == [(2048, 209715)]
    assert initial["reference_export"] == LOCAL_EXPORT_POLICY
    mnist.validate_radius_initialization(initial)
    initial["reference_export"]["reduction"] = "unproven"
    with pytest.raises(ValueError, match="bounded local scale export"):
        mnist.validate_radius_initialization(initial)


def test_local_export_budget_rejects_before_materializing_a_large_outer_product():
    model, _, _, _ = site()
    with pytest.raises(ValueError, match="scratch budget"):
        local_dense_weight(model, maximum_outer_product_values=1)
