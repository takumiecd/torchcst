"""Cold registry Plan overrides preserve frozen radius cohorts and dense controls."""

import copy
import hashlib
import json
import sys
from dataclasses import replace

import pytest
import torch
from test_radius_mnist import initialization as initialization
from test_radius_mnist_1024 import wiring_initialization as wiring_initialization

from benchmarks.training import cell_mass_mnist as mnist
from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.plans import (
    GLOBAL_BATCHED_RESPONSE4_GRAD4,
)


@pytest.mark.parametrize("large", [False, True])
def test_explicit_plan_selects_only_cst_and_keeps_all_frozen_parameters(request, large):
    initial = request.getfixturevalue(
        "wiring_initialization" if large else "initialization"
    )
    frozen = copy.deepcopy(initial)
    baseline_flags = {"radius_execution": "global-response4"} if large else {}
    baseline = mnist.make_model(initial, "cst", **baseline_flags)
    plan = GLOBAL_BATCHED_RESPONSE4_GRAD4
    candidate = mnist.make_model(initial, "cst", execution_plan=plan)
    dense = mnist.make_model(initial, "dense", **baseline_flags)
    dense_override = mnist.make_model(initial, "dense", execution_plan=plan)
    assert candidate[2].selector._plan is plan
    assert baseline[2].selector._plan != plan
    assert type(dense_override[2]) is torch.nn.Linear
    assert torch.equal(dense_override[2].weight, initial["weight"])
    for left, right in ((baseline, candidate), (dense, dense_override)):
        for before, after in zip(left.parameters(), right.parameters()):
            assert torch.equal(before, after)
    assert initial["linear_plan"] == frozen["linear_plan"]
    assert initial["initial_hashes"] == frozen["initial_hashes"]
    assert torch.equal(initial["atoms"], frozen["atoms"])
    assert torch.equal(initial["weight"], frozen["weight"])
    mnist.validate_radius_initialization(initial)


def test_plan_json_roundtrip_and_raw_hash_are_distinct_from_canonical_dump(tmp_path):
    registry = mnist.make_algorithm_registry()
    plan = GLOBAL_BATCHED_RESPONSE4_GRAD4
    payload = json.dumps(registry.dump_plan(plan), indent=3).encode() + b"\n"
    path = tmp_path / "plan.json"
    path.write_bytes(payload)
    actual, digest = mnist.load_radius_execution_plan(path)
    assert actual == plan and digest == hashlib.sha256(payload).hexdigest()
    snapshot = mnist._radius_plan_snapshot(actual, digest)
    assert registry.load_plan(snapshot["plan"]) == plan
    assert snapshot["source_file_sha256"] == digest
    assert (
        snapshot["canonical_sha256"]
        == hashlib.sha256(registry.dumps_plan(plan).encode()).hexdigest()
    )
    assert snapshot["canonical_sha256"] != digest
    assert path.read_bytes() == payload


def test_generic_recipe_fields_are_selected_without_cold_device_support_probe(
    initialization, monkeypatch
):
    from torchcst._backends.cuda.algorithms.linear.radius_cell_mass.algorithm import (
        RadiusCellMassAlgorithm,
    )

    monkeypatch.setattr(
        RadiusCellMassAlgorithm,
        "supports",
        lambda *args: pytest.fail("support belongs to actual first dispatch"),
    )
    plan = replace(
        GLOBAL_BATCHED_RESPONSE4_GRAD4,
        recipe=replace(
            GLOBAL_BATCHED_RESPONSE4_GRAD4.recipe,
            moment_method="direct",
            input_direct_levels=5,
            output_direct_levels=6,
            output_tile_sites=17,
        ),
    )
    model = mnist.make_model(initialization, "cst", execution_plan=plan)
    assert model[2].selector._plan is plan
    assert torch.equal(model[2].atoms.p, initialization["atoms"])
    assert initialization["linear_plan"]["recipe"]["input_direct_levels"] == 2


@pytest.mark.parametrize("change", ["duplicate_key", "unknown_field", "native_global"])
def test_plan_file_decoder_rejects_invalid_json_or_recipe(tmp_path, change):
    registry = mnist.make_algorithm_registry()
    payload = registry.dump_plan(GLOBAL_BATCHED_RESPONSE4_GRAD4)
    if change == "unknown_field":
        payload["recipe"]["unregistered_option"] = True
    elif change == "native_global":
        payload["recipe"]["native_autograd"] = True
    text = json.dumps(payload)
    if change == "duplicate_key":
        text = text.replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    path = tmp_path / "bad-plan.json"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError)):
        mnist.load_radius_execution_plan(path)


@pytest.mark.parametrize(
    "large,kind", [(False, "cst"), (False, "dense"), (True, "cst"), (True, "dense")]
)
def test_valid_plan_snapshot_precedes_data_and_large_keeps_explicit_b4_gate(
    request, large, kind, monkeypatch, tmp_path
):
    initial = request.getfixturevalue(
        "wiring_initialization" if large else "initialization"
    )
    plan = GLOBAL_BATCHED_RESPONSE4_GRAD4
    source_hash = "a" * 64

    def stop(_):
        raise RuntimeError("data boundary reached")

    monkeypatch.setattr(mnist, "load_mnist", stop)
    kwargs = dict(
        batch_size=4, execution_plan=plan, execution_plan_source_sha256=source_hash
    )
    if large:
        with pytest.raises(ValueError, match="allow_radius_1024"):
            mnist.run_training(initial, kind, tmp_path, tmp_path / "out", **kwargs)
        assert not (tmp_path / "out").exists()
        kwargs["allow_radius_1024"] = True
        with pytest.raises(ValueError, match="true batch 4 only"):
            mnist.run_training(
                initial,
                kind,
                tmp_path,
                tmp_path / "out",
                **(kwargs | {"batch_size": 32, "allow_radius_batch32": True}),
            )
        assert not (tmp_path / "out").exists()
    with pytest.raises(RuntimeError, match="data boundary reached"):
        mnist.run_training(initial, kind, tmp_path, tmp_path / "out", **kwargs)
    snapshot = json.loads((tmp_path / "out/execution-plan-snapshot.json").read_text())
    assert snapshot["source_file_sha256"] == source_hash
    assert mnist.make_algorithm_registry().load_plan(snapshot["plan"]) == plan


@pytest.mark.parametrize(
    "change",
    [
        "algorithm",
        "revision",
        "recipe_type",
        "unknown_schedule",
        "native_global",
        "save_response",
        "tile_type",
        "not_plan",
    ],
)
def test_bad_explicit_plan_rejects_before_data_or_output(
    initialization, monkeypatch, tmp_path, change
):
    plan = GLOBAL_BATCHED_RESPONSE4_GRAD4
    if change == "algorithm":
        plan = replace(plan, algorithm_id="torch_factored")
    elif change == "revision":
        plan = replace(plan, algorithm_revision="unknown")
    elif change == "recipe_type":
        plan = replace(plan, recipe={})
    elif change == "unknown_schedule":
        plan = replace(plan, recipe=replace(plan.recipe, schedule="unknown"))
    elif change == "native_global":
        plan = replace(
            plan, recipe=replace(plan.recipe, native_autograd=True, native_entry=True)
        )
    elif change == "save_response":
        plan = replace(plan, recipe=replace(plan.recipe, save_response=False))
    elif change == "tile_type":
        plan = replace(plan, recipe=replace(plan.recipe, output_tile_sites=True))
    else:
        plan = object()
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load data")
    )
    for kind in ("cst", "dense"):
        with pytest.raises(ValueError):
            mnist.make_model(initialization, kind, execution_plan=plan)
        with pytest.raises(ValueError):
            mnist.run_training(
                initialization,
                kind,
                tmp_path,
                tmp_path / "out",
                batch_size=4,
                execution_plan=plan,
            )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "flags",
    [
        {"radius_execution": "batched"},
        {"radius_execution": "global-response4"},
        {"radius_direct_levels": 5},
    ],
)
def test_explicit_plan_requires_default_legacy_flags_before_data(
    initialization, monkeypatch, tmp_path, flags
):
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load data")
    )
    for kind in ("cst", "dense"):
        with pytest.raises(ValueError, match="requires default"):
            mnist.make_model(
                initialization,
                kind,
                execution_plan=GLOBAL_BATCHED_RESPONSE4_GRAD4,
                **flags,
            )
        with pytest.raises(ValueError, match="requires default"):
            mnist.run_training(
                initialization,
                kind,
                tmp_path,
                tmp_path / "out",
                batch_size=4,
                execution_plan=GLOBAL_BATCHED_RESPONSE4_GRAD4,
                **flags,
            )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "legacy", [{"width": 128}, {"width": 128, "parameterization": "polar"}]
)
def test_explicit_radius_plan_rejects_legacy_sigma_initialization_before_data(
    monkeypatch, tmp_path, legacy
):
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load data")
    )
    with pytest.raises(ValueError, match="require a radius initialization"):
        mnist.make_model(legacy, "dense", execution_plan=GLOBAL_BATCHED_RESPONSE4_GRAD4)
    with pytest.raises(ValueError, match="require a radius initialization"):
        mnist.run_training(
            legacy,
            "dense",
            tmp_path,
            tmp_path / "out",
            execution_plan=GLOBAL_BATCHED_RESPONSE4_GRAD4,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("source_hash", ["bad", "A" * 64, 7])
def test_invalid_plan_provenance_rejects_before_data(
    initialization, monkeypatch, tmp_path, source_hash
):
    monkeypatch.setattr(
        mnist, "load_mnist", lambda _: pytest.fail("must not load data")
    )
    with pytest.raises(ValueError, match="SHA256"):
        mnist.run_training(
            initialization,
            "cst",
            tmp_path,
            tmp_path / "out",
            execution_plan=GLOBAL_BATCHED_RESPONSE4_GRAD4,
            execution_plan_source_sha256=source_hash,
        )
    assert not (tmp_path / "out").exists()


def test_cli_loads_registered_plan_once_and_forwards_hash_without_changing_artifact(
    initialization, monkeypatch, tmp_path
):
    registry = mnist.make_algorithm_registry()
    artifact = tmp_path / "initial.pt"
    torch.save(initialization, artifact)
    artifact_bytes = artifact.read_bytes()
    payload = registry.dumps_plan(GLOBAL_BATCHED_RESPONSE4_GRAD4).encode() + b"\n"
    path = tmp_path / "plan.json"
    path.write_bytes(payload)
    seen = []

    def capture(initial, kind, data_directory, output, **kwargs):
        assert initial["linear_plan"] == initialization["linear_plan"]
        assert initial["initial_hashes"] == initialization["initial_hashes"]
        seen.append(kwargs)

    monkeypatch.setattr(mnist, "run_training", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mnist",
            "--initialization",
            str(artifact),
            "--kind",
            "cst",
            "--data-directory",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--radius-plan",
            str(path),
        ],
    )
    mnist.main()
    assert len(seen) == 1
    assert seen[0]["execution_plan"] == GLOBAL_BATCHED_RESPONSE4_GRAD4
    assert (
        seen[0]["execution_plan_source_sha256"] == hashlib.sha256(payload).hexdigest()
    )
    assert (
        seen[0]["radius_execution"] == "cached" and seen[0]["radius_direct_levels"] == 2
    )
    assert artifact.read_bytes() == artifact_bytes
    assert not (tmp_path / "out").exists()
