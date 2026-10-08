"""Independent contracts for the existing Sphere measurement, not a new kernel."""

import pytest
import torch

from benchmarks.cuda.linear.sphere_baseline import (
    correctness,
    fixture,
    forward,
    trajectory_gate,
)


@pytest.mark.parametrize("sigma", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("route", ["factored", "factor-weight", "materialized"])
def test_existing_sphere_matches_full_physical_oracle(sigma, route):
    model, x, _, dy = fixture(17, sigma, batch=3, atoms=15)
    assert correctness(model, x, dy, route)["status"] == "PASS"


def test_existing_sphere_twenty_steps_match_weight_contraction():
    assert trajectory_gate()["status"] == "PASS"


def test_sphere_fixture_is_reproducible_and_has_spherical_centers():
    from torchcst._backends.torch.geometry import execution

    model, x, _, _ = fixture(32, 3.0)
    again, xx, _, _ = fixture(32, 3.0)
    assert torch.equal(model.atoms.p, again.atoms.p)
    assert torch.equal(x, xx)
    for block, chart in zip((slice(2, 4), slice(4, 6)), model.cst_charts()):
        decoded = execution.decode_centers(chart.geometry, model.atoms.p[:, block])
        torch.testing.assert_close(
            decoded.norm(dim=-1), chart.geometry.radius.expand(model.atom_count)
        )
    with pytest.raises(ValueError):
        forward(model, x, "unknown")
