"""CPU qualification of the existing runner's public radius fixture and oracle."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from benchmarks.cuda.linear.manifest import load_run
from benchmarks.cuda.linear.radius_cell_mass import distribution, fixture, oracle

ROOT = Path(__file__).resolve().parents[1]


def case():
    return load_run(
        ROOT / "benchmarks/cuda/linear/cases/radius-cell-mass-128-initial.json",
        ROOT / "benchmarks/cuda/linear/plans-radius-cell-mass.json",
    )


def test_radius_catalog_and_public_initializer():
    run = case()
    model, x, target, _ = fixture(run.case)
    assert model.atoms.p.shape == (819, 4)
    assert x.shape == target.shape == (4, 128)
    assert not run.case.optimizer.capturable
    report = distribution(model)
    assert len(report) == 2
    assert len(report[0]["radii"]) == 819
    assert sum(report[0]["level_counts"]) == 819
    assert sum(report[0]["bank_counts"]) == 819


def test_radius_quadrature_oracle_all_gradients():
    small = replace(case().case, atoms=8)
    model, x, _, dy = fixture(small)
    model = model.double()
    x = x.double().requires_grad_()
    actual = model(x)
    ga = torch.autograd.grad(actual, (x, model.atoms.p), dy.double())
    expected = oracle(model, x)
    gb = torch.autograd.grad(expected, (x, model.atoms.p), dy.double())
    for a, b in zip((actual, *ga), (expected, *gb)):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-8)


def test_radius_protocol_reconstructs_revision7_contract():
    from benchmarks.cuda.linear.protocol import measurement_operator
    from torchcst.kernels.parameterizations import PolarAmpRadiusSpec

    run = case()
    expected, *_ = fixture(run.case)
    actual = measurement_operator(asdict(run.case))
    assert actual == expected.execution_declaration()
    assert actual.kernel.revision == 7
    assert isinstance(actual.kernel.parameterization, PolarAmpRadiusSpec)


def test_radius_submission_adapter_rejects_research_only():
    from benchmarks.database.adapters.linear import project

    artifact = {
        "schema_version": 1,
        "status": "PASS",
        "records": [],
        "run": case().snapshot(),
    }
    with pytest.raises(ValueError, match="polar_radius_cell_mass is research-only"):
        project(artifact)


def test_training_fixture_decays_amplitude_instead_of_coordinates():
    from benchmarks.cuda.linear.radius_cell_mass import optimizer

    run = case()
    model, *_ = fixture(run.case)
    opt = optimizer(model, run.case, gpu=False)
    assert opt.param_groups[0]["weight_decay"] == 0.0
    assert (
        opt.param_groups[0]["amplitude_weight_decay"] == run.case.optimizer.weight_decay
    )
