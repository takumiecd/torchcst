"""Retained-state and independent validation-selection audit regressions."""

import hashlib
import io
import json
import shutil
import tarfile

import numpy as np
import pytest
import torch

from benchmarks.training.graph_higgs_cohort import HIGGS_CONFIRMATION_COHORT
from benchmarks.training.higgs_confirmation_verify import (
    checkpoint_inventory,
    metric_checks,
    prediction_metrics,
    verify_cell,
    verify_evaluation,
    verify_oracle,
    verify_predictions,
    verify_qualification_pairs,
)
from benchmarks.training.higgs_verify import file_sha256, tensor_fields, tensor_hash


def write(path, value):
    path.write_text(json.dumps(value))


def test_checkpoint_inventory_reads_recovery_and_related_bytes(tmp_path):
    def archive_bytes(members):
        target = io.BytesIO()
        with tarfile.open(fileobj=target, mode="w:gz") as archive:
            for name, raw in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
        return target.getvalue()

    job = tmp_path / "job"
    artifacts = job / "results/artifacts"
    artifacts.mkdir(parents=True)
    first, second, third = (
        b"recovered checkpoint",
        b"current checkpoint",
        b"earlier checkpoint",
    )
    recovery = archive_bytes({"confirmed/1024/29/dense/best-dev.pt": first})
    (job / "source.tar.gz").write_bytes(
        archive_bytes({"research-inputs/recovery-state.tar.gz": recovery})
    )
    (artifacts / "final.pt").write_bytes(second)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "best-dev.pt").write_bytes(third)
    inventory = checkpoint_inventory(job, [earlier])
    assert set(inventory) == {
        hashlib.sha256(value).hexdigest() for value in (first, second, third)
    }
    assert (
        "recovery-state.tar.gz!confirmed"
        in inventory[hashlib.sha256(first).hexdigest()][0]
    )
    (earlier / "best-dev.pt").write_bytes(b"changed")
    assert hashlib.sha256(third).hexdigest() not in checkpoint_inventory(job, [earlier])


@pytest.fixture
def cell(tmp_path):
    directory = tmp_path / "synthetic-cell"
    directory.mkdir()
    initial = {
        "0.weight": torch.zeros(1024, 28),
        "0.bias": torch.zeros(1024),
        "2.weight": torch.zeros(1024, 1024),
        "4.weight": torch.zeros(2, 1024),
        "4.bias": torch.zeros(2),
    }
    args = {
        "kind": "dense",
        "width": 1024,
        "divisor": 1000,
        "seed": 29,
        "lr": 0.001,
        "mode": "graph",
        "epochs": 2,
        "steps_per_epoch": 16,
        "development_rows": 4,
    }
    torch.save({"model": initial, "args": args}, directory / "initialization.pt")
    configuration = {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        **args,
        "dataset_sha256": "data",
    }
    epoch_rows = []
    for epoch in (1, 2):
        model = {name: value + epoch * 0.01 for name, value in initial.items()}
        names = ["0.weight", "0.bias", "4.weight", "4.bias", "2.weight"]
        optimizer = {
            "state": {
                index: {
                    "step": torch.tensor(float(epoch * 16)),
                    "exp_avg": torch.ones_like(model[name]),
                    "exp_avg_sq": torch.ones_like(model[name]),
                }
                for index, name in enumerate(names)
            },
            "param_groups": [{"params": [0, 1, 2, 3]}, {"params": [4]}],
        }
        gradients = {
            name: torch.full_like(value, 0.25) for name, value in model.items()
        }
        logits = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32) / epoch
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        metrics = prediction_metrics(logits, labels)
        checkpoint = {
            "model": model,
            "optimizer": optimizer,
            "gradients": gradients,
            "configuration": configuration,
            "epoch": epoch,
            "updates": epoch * 16,
            "development": metrics,
        }
        torch.save(checkpoint, directory / f"epoch-{epoch}.pt")
        row = {
            "epoch": epoch,
            "updates": epoch * 16,
            "development": metrics,
            "state": {
                name: tensor_hash(value)
                for name, value in tensor_fields(
                    {"model": model, "optimizer": optimizer}
                )
            },
            "gradients": {
                name: tensor_hash(value) for name, value in gradients.items()
            },
            "order_sha256": "order-" + str(epoch),
        }
        epoch_rows.append(row)
        if epoch == 1:
            torch.save(checkpoint, directory / "best-dev.pt")
            np.savez_compressed(
                directory / "best-dev-predictions.npz", logits=logits, labels=labels
            )
    best = {
        "epoch": 1,
        "updates": 16,
        "development": epoch_rows[0]["development"],
        "checkpoint_sha256": file_sha256(directory / "best-dev.pt"),
        "predictions_sha256": file_sha256(directory / "best-dev-predictions.npz"),
        "state": epoch_rows[0]["state"],
        "gradients": epoch_rows[0]["gradients"],
    }
    write(directory / "best-dev.json", best)
    learning = args | {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        "alpha": 0.001,
        "trainable_parameters": 1024 * 1024 + 31 * 1024 + 2,
        "dataset_sha256": "data",
        "initialization_sha256": file_sha256(directory / "initialization.pt"),
        "epochs": epoch_rows,
        "final_checkpoint": "epoch-2.pt",
        "best_development_checkpoint": best,
        "development_rows": 4,
        "training_rows": 100000,
        "batch_size": 4,
        "all_layers_trainable": True,
        "TF32": False,
        "cpu_math_threads": 1,
        "total_updates": 32,
        "graph_replays": 24,
        "validation_partition": {
            "development": [0, 4],
            "confirmation": [50000, 100000],
        },
        "confirmation_evaluated": False,
        "test_evaluated": False,
    }
    write(directory / "learning.json", learning)
    return directory


def test_selected_checkpoint_and_final_actual_tensors_verify(cell):
    report = verify_cell(cell, "data", np.arange(100000, dtype=np.int64) % 2)
    assert report["pass"], report["checks"]
    assert report["best_epoch"] == 1
    assert len(report["checkpoints"]) == 3


@pytest.mark.parametrize(
    "change", ("gradient", "clock", "configuration", "best_epoch", "prediction_labels")
)
def test_tampered_state_or_selection_is_rejected(cell, change):
    if change in ("gradient", "clock", "configuration"):
        path = cell / "epoch-2.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if change == "gradient":
            checkpoint["gradients"]["4.bias"][0] += 1
        elif change == "clock":
            checkpoint["optimizer"]["state"][0]["step"].sub_(1)
        else:
            checkpoint["configuration"]["seed"] = 43
        torch.save(checkpoint, path)
    elif change == "best_epoch":
        path = cell / "best-dev.json"
        best = json.loads(path.read_text())
        best["epoch"] = 2
        write(path, best)
    else:
        path = cell / "best-dev-predictions.npz"
        with np.load(path) as data:
            logits, labels = data["logits"].copy(), data["labels"].copy()
        labels[0] = 1 - labels[0]
        np.savez_compressed(path, logits=logits, labels=labels)
    report = verify_cell(cell, "data", np.arange(100000, dtype=np.int64) % 2)
    assert not report["pass"]


def test_actual_qualification_epoch_comparisons_reject_difference(cell, tmp_path):
    eager = tmp_path / "eager"
    shutil.copytree(cell, eager)
    data = json.loads((eager / "learning.json").read_text())
    data["mode"] = "eager"
    write(eager / "learning.json", data)
    paths = [cell / "learning.json", eager / "learning.json"]
    report = verify_qualification_pairs(paths)
    assert report["pass"] and len(report["comparisons"]) == 2
    checkpoint = torch.load(eager / "epoch-2.pt", map_location="cpu", weights_only=True)
    checkpoint["model"]["4.bias"][0] += 0.01
    torch.save(checkpoint, eager / "epoch-2.pt")
    assert not verify_qualification_pairs(paths)["pass"]


def test_prediction_metric_recomputation_and_source_label_order(tmp_path):
    path = tmp_path / "predictions.npz"
    logits = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    metrics = prediction_metrics(logits, labels)
    np.savez_compressed(path, logits=logits, labels=labels)
    assert verify_predictions(path, labels, metrics)["pass"]
    assert not verify_predictions(path, labels[::-1], metrics)["pass"]
    assert not all(metric_checks(metrics, metrics | {"roc_auc": 0.5}).values())


def test_confirmation_checkpoint_sha_and_partition_are_audited(tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    labels = np.arange(100000, dtype=np.int64) % 2
    logits = np.tile(np.array([[1, 0], [0, 1]], dtype=np.float32), (25000, 1))
    target = labels[50000:]
    np.savez_compressed(directory / "predictions.npz", logits=logits, labels=target)
    metrics = prediction_metrics(logits, target)
    import hashlib

    cfg = {
        "cohort": HIGGS_CONFIRMATION_COHORT,
        "kind": "cst",
        "width": 1024,
        "divisor": 1000,
        "seed": 29,
        "lr": 0.001,
        "dataset_sha256": "data",
    }
    metadata = cfg | {
        "checkpoint_configuration": cfg,
        "checkpoint_sha256": "checkpoint",
        "validation_partition": [50000, 100000],
        "confirmation": metrics,
        "predictions_sha256": file_sha256(directory / "predictions.npz"),
        "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "test_evaluated": False,
    }
    write(directory / "evaluation.json", metadata)
    assert verify_evaluation(
        directory / "evaluation.json", labels, "data", {"checkpoint"}
    )["pass"]
    assert not verify_evaluation(
        directory / "evaluation.json", labels, "data", {"other"}
    )["pass"]


def test_retained_oracle_values_are_recomputed(tmp_path):
    path = tmp_path / "N1024-D1000-vjp.pt"
    expected = (torch.ones(4, 1024), torch.ones(4, 1024), torch.ones(1048, 4))
    values = {"actual": expected, "expected": expected, "P": torch.ones(1048, 4)}
    torch.save(values, path)
    assert verify_oracle(path)["pass"]
    actual = [value.clone() for value in expected]
    actual[2][0, 0] = 0.9
    torch.save(values | {"actual": actual}, path)
    assert not verify_oracle(path)["pass"]
