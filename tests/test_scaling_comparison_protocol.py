"""Common inputs, exact oracles, frozen metadata and declared state trajectories."""

import subprocess

import pytest
import torch

from benchmarks.cuda.linear import scaling_comparison as protocol
from benchmarks.cuda.linear.manifest import REGISTRY


@pytest.mark.parametrize("sigma", [3, 8])
def test_geometry_initialization_does_not_change_shared_task(sigma):
    a, *ax = protocol.fixture("sphere", 1024, sigma, batch=2, atoms=7)
    b, *bx = protocol.fixture("torus", 1024, sigma, batch=2, atoms=7)
    assert a.atoms.p.shape == (7, 6)
    assert b.atoms.p.shape == (7, 5)
    for left, right in zip(ax, bx, strict=True):
        assert torch.equal(left, right)
    # Global RNG changes cannot change the declared task or dense fixture.
    torch.manual_seed(99)
    assert torch.equal(
        protocol.dense_fixture(32).weight, protocol.dense_fixture(32).weight
    )
    for actual, expected in zip(ax, protocol.shared_inputs(1024, batch=2), strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("geometry", ["sphere", "torus"])
@pytest.mark.parametrize("sigma", [3, 8])
def test_conditional_baseline_is_registered_and_roundtrips(geometry, sigma):
    plan = protocol.baseline_plan(geometry, sigma)
    assert REGISTRY.loads_plan(REGISTRY.dumps_plan(plan)) == plan
    if geometry == "sphere":
        expected = (
            "research_cuda_sphere_polar_fused_weight"
            if sigma == 3
            else "research_cuda_sphere_polar_blocked"
        )
        assert plan.algorithm_id == expected


def test_oracle_checks_entire_gradient_and_rejects_changed_operator(monkeypatch):
    model, x, target, dy = protocol.fixture("sphere", 32, 3, batch=2, atoms=7)
    model.selector = None
    report = protocol.oracle_check(model, x, dy, target, "sphere")
    assert report["status"] == "PASS"
    assert set(report["errors"]) == {"Y", "dX", "all_dP"}
    original = model.forward
    monkeypatch.setattr(model, "forward", lambda value: original(value) + 0.01)
    with pytest.raises(AssertionError):
        protocol.oracle_check(model, x, dy, target, "sphere")


def test_archive_without_git_records_hashes_and_separate_hint(monkeypatch):
    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(protocol.subprocess, "check_output", no_git)
    monkeypatch.setenv("CST_FROZEN_SOURCE_COMMIT", "a" * 40)
    metadata = protocol.source_metadata()
    assert metadata["source_commit"] is None
    assert metadata["source_commit_hint"] == "a" * 40
    assert (
        "src/torchcst/_backends/cuda/algorithms/linear/torus_profile_product/_shared/profiles.py"
        in metadata["source_hashes"]
    )
    assert "benchmarks/cuda/linear/sphere_baseline.py" in metadata["source_hashes"]
    assert (
        "benchmarks/cuda/linear/torus_profile_product.py" in metadata["source_hashes"]
    )


def test_zero_amplitude_support_is_not_reported_empty():
    model, *_ = protocol.fixture("sphere", 32, 3, batch=2, atoms=7)
    with torch.no_grad():
        model.atoms.p[:, 0] = 0
        model.atoms.p[:, 2:] = 0
    summary = protocol.support_summary(
        model, "sphere", protocol.baseline_plan("sphere", 3)
    )
    assert all(side["max_count"] > 0 for side in summary["sides"])
    assert all(side["empty_atoms"] == 0 for side in summary["sides"])
    assert len(protocol.fixture_metadata(model)["declaration_sha256"]) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
@pytest.mark.parametrize("geometry", ["sphere", "torus"])
@pytest.mark.parametrize("sigma", [3, 8])
def test_same_cotangent_public_parameters_moments_and_exact_clock(geometry, sigma):
    proof = protocol.state_trajectory(geometry, sigma)
    assert proof["status"] == "PASS"
    assert proof["same_cotangent"] and proof["exact_step_counter"]
