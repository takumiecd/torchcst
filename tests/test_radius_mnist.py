"""Frozen radius MNIST tooling stays matched without loading MNIST or CUDA."""

import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.cuda.linear.radius_cell_mass import fixture
from benchmarks.training import cell_mass_mnist as mnist
from torchcst import CSTAdamW
from torchcst._backends.cuda.algorithms.polar_update.plans import (
    RADIUS_FLAT_TORUS_FUSED,
)


@pytest.fixture(scope="module")
def initialization():
    return mnist.make_radius_initialization()


def test_radius_initialization_matches_public_fixture_and_catalog(initialization):
    reference, *_ = fixture(SimpleNamespace(seed=41, size=128, atoms=819, rows=4))
    assert torch.equal(initialization["atoms"], reference.atoms.p)
    assert torch.equal(initialization["weight"], reference.dense_weight())
    assert initialization["kernel_settings"] == mnist.radius_settings()
    catalog = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/cuda/linear/plans-radius-cell-mass.json"
        ).read_text()
    )
    canonical = next(
        row["plan"]
        for row in catalog["plans"]
        if row["id"] == "radius-native-cached-parallel"
    )
    assert initialization["linear_plan"] == canonical
    assert initialization["cohort"] == mnist.RADIUS_COHORT
    assert initialization["parameterization"] == "polar_radius"
    assert "sigma_initial" not in initialization
    assert initialization["amplitude_gain"] == 1.0
    assert initialization["kernel_buffers_fp32"] == mnist.buffer_metadata(
        reference.kernel
    )
    assert initialization["kernel_buffers_fp32"]["w_c"]["values"] == [
        float(torch.tensor(0.3, dtype=torch.float32))
    ]


def test_radius_artifact_roundtrip_preserves_weights_outer_states_and_order(
    initialization,
):
    stream = io.BytesIO()
    torch.save(initialization, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    cst, dense = mnist.make_model(restored, "cst"), mnist.make_model(restored, "dense")
    assert torch.equal(cst[2].dense_weight(), dense[2].weight)
    for index in (0, 4):
        for name, value in cst[index].state_dict().items():
            assert torch.equal(value, dense[index].state_dict()[name])
    for epoch in range(3):
        left = torch.randperm(
            97,
            generator=torch.Generator().manual_seed(
                initialization["seed"] + 100 + epoch
            ),
        )
        right = torch.randperm(
            97, generator=torch.Generator().manual_seed(restored["seed"] + 100 + epoch)
        )
        assert torch.equal(left, right)
    # The CPU reference uses the ordinary Torch route for this numerical gate.
    cst[2].selector = None
    cst[2].backend = "factored"
    inputs = torch.randn(4, 784)
    torch.testing.assert_close(cst(inputs), dense(inputs), atol=2e-6, rtol=2e-6)


def test_radius_optimizer_matches_ordinary_groups_and_keeps_atom_decay_logical(
    initialization,
):
    cst, dense = (
        mnist.make_model(initialization, "cst"),
        mnist.make_model(initialization, "dense"),
    )
    actual = mnist.make_optimizer(cst, initialization, "cst")
    control = mnist.make_optimizer(dense, initialization, "dense")
    assert type(actual) is CSTAdamW
    assert type(control) is torch.optim.AdamW
    assert actual.check_finite is False
    assert not actual.state and not control.state  # No hidden update warmup.
    assert actual._update_dispatcher.selector._plan == RADIUS_FLAT_TORUS_FUSED
    for opt in (actual, control):
        assert opt.param_groups[0]["role"] == "ordinary"
        assert opt.param_groups[0]["weight_decay"] == 0.01
        assert all(
            group["lr"] == 1e-4 and group["fused"] and not group["capturable"]
            for group in opt.param_groups
        )
    assert actual.param_groups[1]["weight_decay"] == 0.0
    assert actual.param_groups[1]["amplitude_weight_decay"] == 0.01
    assert control.param_groups[1]["weight_decay"] == 0.01
    assert "amplitude_weight_decay" not in control.param_groups[1]
    ordinary_ids = {id(p) for p in (*cst[0].parameters(), *cst[4].parameters())}
    assert {id(p) for p in actual.param_groups[0]["params"]} == ordinary_ids
    assert actual.param_groups[1]["params"] == [cst[2].atoms.p]


def test_radius_diagnostics_cover_all_atoms_without_sigma_fields(initialization):
    stats = mnist.radius_statistics(mnist.make_model(initialization, "cst")[2])
    assert stats["valid"] and stats["units"] == "physical support radius"
    assert "sigma" not in json.dumps(stats)
    for side in ("input", "output"):
        assert sum(stats[side]["owner_counts"]) == 819
        assert sum(bank["atoms"] for bank in stats[side]["banks"]) == 819
        assert stats[side]["spacing"] == 1.0
        assert stats[side]["chart_period"] == 128.0
        assert (
            stats[side]["radius_quantiles"]
            == initialization["radius_initial"][side]["radius_quantiles"]
        )


@pytest.mark.parametrize(
    "change", ["atoms", "weight", "input", "settings", "buffers", "optimizer", "plan"]
)
def test_changed_radius_artifact_rejects_before_training(initialization, change):
    changed = copy.deepcopy(initialization)
    if change in ("atoms", "weight"):
        changed[change].flatten()[0] += 0.1
    elif change == "input":
        changed["input"]["weight"].flatten()[0] += 0.1
    elif change == "settings":
        changed["kernel_settings"]["radius_bounds"]["minimum"] = 0.25
    elif change == "buffers":
        changed["kernel_buffers_fp32"]["w_c"]["values"][0] = 0.4
    elif change == "optimizer":
        changed["optimizer_contract"]["amplitude_weight_decay"] = 0.0
    else:
        changed["linear_plan"]["recipe"]["native_entry"] = False
    with pytest.raises(ValueError, match="radius initialization"):
        mnist.make_model(changed, "cst")


@pytest.mark.parametrize("batch,allowed", [(4, False), (32, True)])
def test_radius_batch_policy_reaches_data_only_after_explicit_gate(
    initialization, monkeypatch, tmp_path, batch, allowed
):
    def stop_before_data(_):
        raise RuntimeError("data boundary reached; no training")

    monkeypatch.setattr(mnist, "load_mnist", stop_before_data)
    with pytest.raises(RuntimeError, match="data boundary reached"):
        mnist.run_training(
            initialization,
            "cst",
            tmp_path,
            tmp_path / "out",
            batch_size=batch,
            allow_radius_batch32=allowed,
        )


@pytest.mark.parametrize("batch", [1, 8, 32, 64])
def test_radius_other_batches_reject_without_data_or_output(
    initialization, monkeypatch, tmp_path, batch
):
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load MNIST")
    )
    with pytest.raises(ValueError, match="true batch 4"):
        mnist.run_training(
            initialization, "dense", tmp_path, tmp_path / "out", batch_size=batch
        )
    assert not (tmp_path / "out").exists()


def test_radius_1024_and_lr_override_remain_explicitly_unsupported(initialization):
    with pytest.raises(ValueError, match="only width 128"):
        mnist.make_radius_initialization(1024)
    cst = mnist.make_model(initialization, "cst")
    with pytest.raises(ValueError, match="frozen initialization contract"):
        mnist.make_optimizer(cst, initialization, "cst", lr=0.001)


def test_radius_initialization_cli_never_loads_data_or_initializes_cuda(tmp_path):
    artifact = tmp_path / "radius.pt"
    code = """
import sys
import torch
from benchmarks.training import cell_mass_mnist as mnist
def forbidden(_):
    raise AssertionError('initialization must not load data')
mnist.load_mnist = forbidden
sys.argv = ['mnist', '--create-radius-initialization', sys.argv[1], '--seed', '41']
assert not torch.cuda.is_initialized()
mnist.main()
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(artifact)],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    restored = torch.load(artifact, map_location="cpu", weights_only=True)
    mnist.validate_radius_initialization(restored)
    assert json.loads(result.stdout)["cohort"] == mnist.RADIUS_COHORT
