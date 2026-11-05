"""CPU integrity and pairing regressions for retained confirmation analysis."""

import hashlib
import json

import numpy as np
import pytest

from benchmarks.training import higgs_confirmation_analysis as analysis


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def assessment(tmp_path):
    directory = tmp_path / "assessment"
    directory.mkdir()
    data = tmp_path / "screen.npz"
    labels = np.arange(100000, dtype=np.int64) % 2
    np.savez_compressed(data, y_val=labels)
    identity = {
        "source_identity": "source",
        "source_code_sha256": "code",
        "protocol_sha256": "protocol",
        "dataset_sha256": analysis.digest(data),
    }
    rows, evaluations = [], []
    for width, seed, kind, divisor in sorted(analysis.expected_cases()):
        config_id = "dense" if kind == "dense" else f"cst-{divisor}"
        config = {"id": config_id, "kind": kind, "width": width, "divisor": divisor}
        relative = f"evaluations/{width}/{seed}/{config_id}"
        target = directory / relative
        target.mkdir(parents=True)
        # Seed-sensitive values prove pairing; cancellation proves FP32 subtraction.
        logits = np.full((50000, 2), np.float32(2**24), dtype=np.float32)
        logits[:, 1] += np.float32(2 * (seed + (kind == "cst")))
        np.savez_compressed(
            target / "predictions.npz", logits=logits, labels=labels[50000:]
        )
        row = {
            "configuration": config,
            "seed": seed,
            "lr": 0.001,
            "horizon": 10,
            "best_epoch": 7,
            "checkpoint_sha256": "checkpoint",
            "best_metadata_sha256": "metadata",
            "learning_sha256": "learning",
            "relative_directory": f"confirmed/{width}/{seed}/{config_id}",
        }
        evaluation = {
            "schema_version": 1,
            "cohort": analysis.COHORT,
            "kind": kind,
            "width": width,
            "divisor": divisor,
            "alpha": 1 / divisor,
            "seed": seed,
            "lr": 0.001,
            "checkpoint_sha256": "checkpoint",
            "checkpoint_epoch": 7,
            "checkpoint_updates": 175000,
            "dataset_sha256": identity["dataset_sha256"],
            "validation_partition": [50000, 100000],
            "confirmation": {"examples": 50000},
            "test_evaluated": False,
            "batch_size": 4,
            "TF32": False,
            "torch": "test",
            "cuda": "test",
            "gpu": "NVIDIA L4",
            "predictions_sha256": analysis.digest(target / "predictions.npz"),
            "labels_sha256": hashlib.sha256(labels[50000:].tobytes()).hexdigest(),
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "checkpoint_configuration": {
                "cohort": analysis.COHORT,
                "kind": kind,
                "width": width,
                "divisor": divisor,
                "seed": seed,
                "lr": 0.001,
                "epochs": 10,
                "dataset_sha256": identity["dataset_sha256"],
            },
        }
        write_json(target / "evaluation.json", evaluation)
        rows.append(row)
        evaluations.append(
            {
                **row,
                "evaluation_directory": relative,
                "evaluation_sha256": analysis.digest(target / "evaluation.json"),
                "predictions_sha256": evaluation["predictions_sha256"],
            }
        )
    freeze = {
        "schema_version": 1,
        "identity": identity,
        "configurations": rows,
        "configuration_count": 30,
        "confirmation_partition": [50000, 100000],
        "test_evaluated": False,
    }
    write_json(directory / "confirmation-freeze.json", freeze)
    freeze_sha = analysis.digest(directory / "confirmation-freeze.json")
    index = {
        "schema_version": 1,
        "identity": identity,
        "confirmation_freeze_sha256": freeze_sha,
        "assessments": [
            {**row, "confirmation_freeze_sha256": freeze_sha} for row in evaluations
        ],
        "confirmation_evaluated": True,
        "test_evaluated": False,
    }
    # Unsorted retained rows must still pair by declared seed, never index order.
    index["assessments"].reverse()
    write_json(directory / "assess-index.json", index)
    return directory, data, index


def test_complete_verification_and_eight_paired_comparisons(
    assessment, monkeypatch, tmp_path
):
    directory, data, _ = assessment
    calls = []
    real_compare = analysis.compare_predictions

    def compare(cst, dense, labels, *, width, alpha):
        assert cst.dtype == dense.dtype == np.float32
        assert cst.shape == dense.shape == (3, 50000)
        assert np.array_equal(dense[:, 0], np.array(analysis.CONFIRMATION_SEEDS) * 2)
        assert np.array_equal(
            cst[:, 0], (np.array(analysis.CONFIRMATION_SEEDS) + 1) * 2
        )
        assert np.array_equal(labels, np.arange(50000) % 2)
        calls.append((width, alpha))
        return real_compare(
            cst, dense, labels, width=width, alpha=alpha, bootstrap_replicates=20
        )

    monkeypatch.setattr(analysis, "compare_predictions", compare)
    output = tmp_path / "analysis"
    result = analysis.main(
        ["--assessment", str(directory), "--data", str(data), "--output", str(output)]
    )
    assert len(calls) == 8
    assert len(set(calls)) == 8
    assert len(result["verified_cases"]) == 30
    assert len(result["minimum_alpha_summary"]["widths"]) == 2
    assert all(row["status"] == "supported" for row in result["comparisons"])
    assert json.loads((output / "analysis.json").read_text()) == result
    assert len((output / "comparisons.csv").read_text().splitlines()) == 9


@pytest.mark.parametrize(
    "change",
    ["duplicate", "missing", "evaluation_bytes", "dataset", "labels", "checkpoint"],
)
def test_rejects_incomplete_or_tampered_evidence(assessment, change):
    directory, data, index = assessment
    row = index["assessments"][0]
    target = directory / row["evaluation_directory"]
    if change == "duplicate":
        index["assessments"][0] = index["assessments"][1]
    elif change == "missing":
        index["assessments"].pop()
    elif change == "evaluation_bytes":
        (target / "evaluation.json").write_text("{}")
    elif change == "dataset":
        np.savez_compressed(data, y_val=np.ones(100000, dtype=np.int64))
    else:
        evaluation = analysis.read_json(target / "evaluation.json")
        if change == "labels":
            with np.load(target / "predictions.npz") as retained:
                logits, labels = retained["logits"], retained["labels"].copy()
            labels[0] = 1 - labels[0]
            np.savez_compressed(
                target / "predictions.npz", logits=logits, labels=labels
            )
            evaluation["labels_sha256"] = hashlib.sha256(labels.tobytes()).hexdigest()
            evaluation["predictions_sha256"] = analysis.digest(
                target / "predictions.npz"
            )
            row["predictions_sha256"] = evaluation["predictions_sha256"]
        else:
            evaluation["checkpoint_configuration"]["seed"] = 17
        write_json(target / "evaluation.json", evaluation)
        row["evaluation_sha256"] = analysis.digest(target / "evaluation.json")
    write_json(directory / "assess-index.json", index)
    with pytest.raises(ValueError):
        analysis.verify_assessment(directory, data)


@pytest.mark.parametrize("relative", ["../outside", "/absolute"])
def test_rejects_unsafe_paths(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe"):
        analysis.checked_path(tmp_path, relative)


def test_fresh_output_required(tmp_path):
    with pytest.raises(FileExistsError):
        analysis.main(
            ["--assessment", "unused", "--data", "unused", "--output", str(tmp_path)]
        )
