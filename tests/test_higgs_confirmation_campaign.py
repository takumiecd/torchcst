"""Development-only selection, immutable freezes and deferred-assessment gates."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.training.higgs_confirmation_campaign import (
    Campaign,
    _require_qualification,
    _validate_cell,
    _write_immutable,
    configurations,
    edge_extension,
    file_sha256,
    late_improvement,
    protocol,
    run,
    select_candidate,
)


def _candidate(rate, loss=0.5, best_epoch=4, horizon=10, improving=False):
    curve = [loss + 0.02] * horizon
    curve[best_epoch - 1] = loss
    if improving:
        curve[-3:] = [loss + 0.003, loss + 0.002, loss]
    return {
        "lr": rate,
        "epochs": horizon,
        "best_epoch": best_epoch,
        "best_development_loss": loss,
        "development_losses": curve,
        "checkpoint_sha256": "checkpoint_fixture",
    }


def test_development_selection_ties_and_unique_rates_are_explicit():
    values = [_candidate(0.001), _candidate(0.0003), _candidate(0.003, 0.6)]
    assert select_candidate(values)["lr"] == 0.0003
    with pytest.raises(ValueError, match="unique"):
        select_candidate([values[0], copy.deepcopy(values[0])])


@pytest.mark.parametrize(
    "rate,expected", [(0.0001, 0.00003), (0.003, 0.01), (0.001, None)]
)
def test_only_a_selected_base_grid_edge_gets_one_extension(rate, expected):
    assert edge_extension(rate) == expected


def test_horizon_requires_both_a_late_best_and_development_improvement():
    assert late_improvement(_candidate(0.001, best_epoch=10, improving=True), 10)
    assert not late_improvement(_candidate(0.001, best_epoch=4, improving=True), 10)
    plateau = _candidate(0.001, best_epoch=10)
    plateau["development_losses"][-3:] = [0.5005, 0.5004, 0.5]
    assert not late_improvement(plateau, 10)
    with pytest.raises(ValueError, match="budget"):
        late_improvement(_candidate(0.001), 20)


def test_frozen_record_cannot_be_silently_changed(tmp_path):
    path = tmp_path / "record.json"
    value = {"selected_lr": 0.001, "confirmation_evaluated": False}
    _write_immutable(path, value)
    _write_immutable(path, value)
    with pytest.raises(ValueError, match="frozen"):
        _write_immutable(path, {"selected_lr": 0.003})
    assert json.loads(path.read_text()) == value


def test_qualification_is_bound_to_current_source_and_dataset(tmp_path):
    identity = {"source_identity": "source", "dataset_sha256": "data"}
    _write_immutable(
        tmp_path / "qualification.json", {"status": "passed", "identity": identity}
    )
    _require_qualification(tmp_path, identity)
    with pytest.raises(ValueError, match="qualification"):
        _require_qualification(tmp_path, {**identity, "dataset_sha256": "changed"})


def test_cpu_manifest_needs_neither_data_nor_gpu(tmp_path, capsys):
    run(SimpleNamespace(manifest=True, stage="tune"))
    value = json.loads(capsys.readouterr().out)
    assert value["stage"] == "tune"
    assert value["protocol"]["confirmation_partition"] == [50000, 100000]
    assert value["protocol"]["primary_configuration_family_size"] == 8
    assert not any(tmp_path.iterdir())


def _campaign(tmp_path, width=1024):
    campaign = Campaign.__new__(Campaign)
    campaign.state = tmp_path / "state"
    campaign.identity = {"source_identity": "source", "dataset_sha256": "data"}
    campaign.args = SimpleNamespace(output=tmp_path / "phase", width=width)
    campaign.args.output.mkdir()
    return campaign


def _write_groups(campaign, improving=False):
    for config in configurations(campaign.args.width):
        selected = _candidate(
            0.001,
            best_epoch=10 if improving and config["kind"] == "dense" else 4,
            improving=improving and config["kind"] == "dense",
        )
        group = {
            "identity": campaign.identity,
            "configuration": config,
            "horizon": 10,
            "selected": selected,
            "candidates": [selected, _candidate(0.0003, 0.51), _candidate(0.003, 0.6)],
            "selected_lr_on_evaluated_boundary": False,
        }
        _write_immutable(
            campaign.state
            / "tuning"
            / str(campaign.args.width)
            / config["id"]
            / "selection.json",
            group,
        )


def test_stable_development_curves_freeze_common_ten_epoch_budget(tmp_path):
    campaign = _campaign(tmp_path)
    _write_groups(campaign)
    campaign.cell = lambda *args: pytest.fail(
        "stable curves must not launch more models"
    )
    result = campaign.freeze_width()["frozen"]
    assert result["horizon"] == 10
    assert len(result["configurations"]) == 5
    assert all(row["horizon"] == 10 for row in result["configurations"])
    assert not result["confirmation_evaluated"]


def test_one_late_group_extends_all_configs_and_reruns_only_top_two_lrs(tmp_path):
    campaign = _campaign(tmp_path)
    _write_groups(campaign, improving=True)
    calls = []

    def cell(label, config, seed, rate, epochs):
        calls.append((config["id"], seed, rate, epochs))
        return _candidate(rate, 0.49 if rate == 0.0003 else 0.5, horizon=20)

    campaign.cell = cell
    frozen = campaign.freeze_width()["frozen"]
    assert frozen["horizon"] == 20
    assert frozen["horizon_extension_trigger_configurations"] == ["dense"]
    assert len(calls) == 10
    assert {row[1] for row in calls} == {17}
    assert {row[2] for row in calls} == {0.0003, 0.001}
    assert {row[3] for row in calls} == {20}
    assert all(
        row["horizon"] == 20 and row["lr"] == 0.0003 for row in frozen["configurations"]
    )
    assert not frozen["confirmation_evaluated"]


def test_assessment_does_not_freeze_or_evaluate_until_all_models_exist(tmp_path):
    campaign = _campaign(tmp_path)
    campaign.record = {}
    campaign.persist = lambda: None
    for width in (1024, 2048):
        frozen = {
            "identity": campaign.identity,
            "width": width,
            "horizon": 10,
            "configurations": [
                {**config, "lr": 0.001} for config in configurations(width)
            ],
        }
        _write_immutable(campaign.state / "frozen" / f"{width}.json", frozen)
    with pytest.raises(FileNotFoundError):
        campaign.assess()
    assert not (campaign.state / "confirmation-freeze.json").exists()
    assert not (campaign.args.output / "evaluations").exists()


def test_checkpoint_hash_tamper_is_rejected_before_deferred_assessment(tmp_path):
    campaign = _campaign(tmp_path)
    directory = campaign.state / "confirmed/1024/29/dense"
    directory.mkdir(parents=True)
    checkpoint = directory / "best-dev.pt"
    checkpoint.write_bytes(b"original checkpoint")
    record = {
        "identity": campaign.identity,
        "configuration": configurations(1024)[0],
        "seed": 29,
        "files": {"best-dev.pt": file_sha256(checkpoint)},
    }
    _write_immutable(directory / "complete.json", record)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign._retained(directory, configurations(1024)[0], 29)


def test_protocol_keeps_confirmation_out_of_selection():
    value = protocol()
    assert value["development_seed"] == 17
    assert value["confirmation_seeds"] == [29, 43, 71]
    assert value["divisors"] == [1000, 500, 200, 100]
    assert value["diagnostic_simultaneous_two_sided_endpoint_family_size"] == 16
    assert value["checkpoint_selection"].startswith("minimum development")
    assert not value["test_evaluated"]


def test_full_cell_inventory_checks_development_selection_and_checkpoint_bytes(
    tmp_path,
):
    config = configurations(1024)[0]
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "best-dev.pt").write_bytes(b"test checkpoint bytes")
    losses = [0.7, 0.6] + [0.65] * 8
    learning = {
        "kind": "dense",
        "width": 1024,
        "divisor": 100,
        "seed": 17,
        "lr": 0.001,
        "training_rows": 100000,
        "development_rows": 50000,
        "batch_size": 4,
        "dataset_sha256": "data",
        "total_updates": 250000,
        "validation_partition": {
            "development": [0, 50000],
            "confirmation": [50000, 100000],
        },
        "epochs": [{"development": {"loss": value}} for value in losses],
        "training_seconds": 1,
        "adoption_seconds": 2,
    }
    best = {
        "epoch": 2,
        "development": {"loss": 0.6, "examples": 50000},
        "checkpoint_sha256": file_sha256(cell / "best-dev.pt"),
    }
    _write_immutable(cell / "learning.json", learning)
    _write_immutable(cell / "best-dev.json", best)
    actual = _validate_cell(cell, config, 17, 0.001, 10, {"dataset_sha256": "data"})
    assert actual["best_epoch"] == 2
    best["epoch"] = 3
    (cell / "best-dev.json").write_text(json.dumps(best))
    with pytest.raises(ValueError, match="development loss"):
        _validate_cell(cell, config, 17, 0.001, 10, {"dataset_sha256": "data"})


def test_all_thirty_checkpoints_freeze_before_first_inference_and_resume_without_rerun(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    campaign.record = {}
    campaign.persist = lambda: None
    campaign.args.data = tmp_path / "data.npz"
    campaign.args.data.write_bytes(b"dataset fixture")
    for width in (1024, 2048):
        frozen = {
            "identity": campaign.identity,
            "width": width,
            "horizon": 10,
            "configurations": [
                {**config, "lr": 0.001} for config in configurations(width)
            ],
        }
        _write_immutable(campaign.state / "frozen" / f"{width}.json", frozen)
        for seed in (29, 43, 71):
            for config in configurations(width):
                cell = tmp_path / "fresh-cells" / str(width) / str(seed) / config["id"]
                cell.mkdir(parents=True)
                for name in ("best-dev.pt", "best-dev.json", "learning.json"):
                    (cell / name).write_text(f"{width} {seed} {config['id']} {name}")
                selected = _candidate(0.001)
                selected["checkpoint_sha256"] = file_sha256(cell / "best-dev.pt")
                campaign._retain_confirmation(
                    cell,
                    campaign.state
                    / "confirmed"
                    / str(width)
                    / str(seed)
                    / config["id"],
                    config,
                    seed,
                    selected,
                )
    calls = []

    def evaluate(command, **kwargs):
        freeze_path = campaign.state / "confirmation-freeze.json"
        freeze = json.loads(freeze_path.read_text())
        assert freeze["configuration_count"] == len(freeze["configurations"]) == 30
        assert (
            campaign.args.output / "confirmation-freeze.json"
        ).read_bytes() == freeze_path.read_bytes()
        checkpoint = Path(command[command.index("--evaluate-only-checkpoint") + 1])
        destination = Path(command[command.index("--evaluation-output") + 1])
        destination.mkdir()
        (destination / "predictions.npz").write_bytes(b"predictions fixture")
        _write_immutable(
            destination / "evaluation.json",
            {
                "checkpoint_sha256": file_sha256(checkpoint),
                "dataset_sha256": campaign.identity["dataset_sha256"],
                "validation_partition": [50000, 100000],
                "checkpoint_epoch": 4,
                "predictions_sha256": file_sha256(destination / "predictions.npz"),
            },
        )
        calls.append(command)

    monkeypatch.setattr(
        "benchmarks.training.higgs_confirmation_campaign.subprocess.run", evaluate
    )
    first = campaign.assess()
    assert len(calls) == len(first["assessments"]) == 30
    assert not first["test_evaluated"]
    campaign.args.output = tmp_path / "resumed-phase"
    campaign.args.output.mkdir()
    second = campaign.assess()
    assert len(calls) == 30
    assert first["confirmation_freeze_sha256"] == second["confirmation_freeze_sha256"]
    assert len(second["assessments"]) == 30
