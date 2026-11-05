"""Read-only integrity and checkpoint-selection audit of confirmation jobs.

This verifies the producer's retained values; it does not turn three training
seeds into a stronger statistical claim. No CUDA execution is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np

from benchmarks.training.graph_higgs_cohort import HIGGS_CONFIRMATION_COHORT
from benchmarks.training.higgs_confirmation import (
    CONFIRMATION_ROWS,
    DEV_ROWS,
    atom_count,
)
from benchmarks.training.higgs_density import binary_auc
from benchmarks.training.higgs_verify import (
    byte_equal,
    file_sha256,
    finite_fields,
    read_json,
    safe_path,
    tensor_fields,
    tensor_hash,
    verify_archive,
    verify_git_source,
)


def prediction_metrics(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.ndim != 2 or logits.shape != (len(labels), 2):
        raise ValueError("expected paired binary logits")
    if logits.dtype != np.float32 or labels.dtype != np.int64:
        raise ValueError("prediction dtype differs from FP32/INT64 protocol")
    if not np.isfinite(logits).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("predictions require finite logits and binary labels")
    values = logits.astype(np.float64)
    loss = (
        np.logaddexp(values[:, 0], values[:, 1])
        - values[np.arange(len(labels)), labels]
    )
    return {
        "loss": float(loss.mean()),
        "accuracy": float((logits.argmax(1) == labels).mean()),
        "roc_auc": binary_auc(logits[:, 1] - logits[:, 0], labels),
        "examples": len(labels),
        "logit_rms": float(np.sqrt(np.square(values).mean())),
    }


def metric_checks(actual, recorded):
    tolerances = {"loss": 3e-6, "accuracy": 1e-7, "roc_auc": 1e-12, "logit_rms": 3e-6}
    return {
        name: abs(actual[name] - recorded[name]) <= tolerance
        for name, tolerance in tolerances.items()
    } | {"examples": actual["examples"] == recorded["examples"]}


def verify_predictions(path, expected_labels, recorded):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"logits", "labels"}:
            raise ValueError("unexpected prediction archive members")
        logits, labels = data["logits"], data["labels"]
        recomputed = prediction_metrics(logits, labels)
        checks = metric_checks(recomputed, recorded)
        checks["source_labels_in_original_order"] = np.array_equal(
            labels, expected_labels
        )
        return {
            "checks": checks,
            "metrics_recomputed": recomputed,
            "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "pass": all(checks.values()),
        }


def verify_checkpoint(path, learning, row, initial):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    fields = dict(
        tensor_fields(
            {"model": checkpoint["model"], "optimizer": checkpoint["optimizer"]}
        )
    )
    gradients = checkpoint["gradients"]
    clocks = {
        str(k): float(v["step"].item())
        for k, v in checkpoint["optimizer"]["state"].items()
    }
    changed = {
        name: not byte_equal(initial["model"][name], checkpoint["model"][name])
        for name in ("0.weight", "0.bias", "4.weight", "4.bias")
    }
    configuration = checkpoint["configuration"]
    groups = checkpoint["optimizer"]["param_groups"]
    owned = [key for group in groups for key in group["params"]]
    checks = {
        "cohort": configuration["cohort"] == HIGGS_CONFIRMATION_COHORT,
        "model_optimizer_hashes": {k: tensor_hash(v) for k, v in fields.items()}
        == row["state"],
        "gradient_hashes": {k: tensor_hash(v) for k, v in gradients.items()}
        == row["gradients"],
        "finite": not finite_fields(fields)
        and not finite_fields(dict(tensor_fields(gradients))),
        "epoch": checkpoint["epoch"] == row["epoch"],
        "updates": checkpoint["updates"] == row["updates"],
        "five_clocks": len(clocks) == 5
        and all(v == row["updates"] for v in clocks.values()),
        "optimizer_ownership": [len(group["params"]) for group in groups] == [4, 1]
        and len(owned) == len(set(owned)) == 5
        and set(owned) == set(checkpoint["optimizer"]["state"]),
        "ordinary_layers_changed": all(changed.values()),
        "configuration": all(
            configuration[k] == learning[k]
            for k in ("kind", "width", "divisor", "seed", "lr", "dataset_sha256")
        ),
        "development_metrics": checkpoint["development"] == row["development"],
    }
    result = {
        "file": path.name,
        "epoch": row["epoch"],
        "updates": row["updates"],
        "tensor_fields": len(fields),
        "gradient_fields": len(gradients),
        "clocks": clocks,
        "ordinary_parameter_changed": changed,
        "checks": checks,
        "pass": all(checks.values()),
    }
    del checkpoint, fields, gradients
    gc.collect()
    return result


def verify_cell(directory, data_sha, source_labels):
    import torch

    learning = read_json(directory / "learning.json")
    initial = torch.load(
        directory / "initialization.pt", map_location="cpu", weights_only=True
    )
    epochs, best = learning["epochs"], read_json(directory / "best-dev.json")
    if not epochs:
        raise ValueError("training cell has no epochs")
    expected_best = min(
        epochs, key=lambda row: (row["development"]["loss"], row["epoch"])
    )
    steps = epochs[0]["updates"]
    selected_row = next(row for row in epochs if row["epoch"] == best["epoch"])
    checkpoints = [
        verify_checkpoint(directory / "best-dev.pt", learning, selected_row, initial)
    ]
    final_path = safe_path(directory, learning["final_checkpoint"])
    checkpoints.append(verify_checkpoint(final_path, learning, epochs[-1], initial))
    seen = {final_path.name, "best-dev.pt"}
    for path in sorted(directory.glob("epoch-*.pt")):
        if path.name not in seen:
            number = int(re.fullmatch(r"epoch-(\d+)\.pt", path.name).group(1))
            row = next(row for row in epochs if row["epoch"] == number)
            checkpoints.append(verify_checkpoint(path, learning, row, initial))
    dev_rows = learning["development_rows"]
    width = learning["width"]
    count = atom_count(width, learning["divisor"])
    model_shapes = {
        name: tuple(value.shape)
        for name, value in initial["model"].items()
        if isinstance(value, torch.Tensor)
    }
    expected_shapes = {
        "0.weight": (width, 28),
        "0.bias": (width,),
        "4.weight": (2, width),
        "4.bias": (2,),
    }
    middle_key, middle_shape = (
        ("2.atom_state.atoms.p", (count, 4))
        if learning["kind"] == "cst"
        else ("2.weight", (width, width))
    )
    expected_shapes[middle_key] = middle_shape
    predictions = verify_predictions(
        directory / "best-dev-predictions.npz",
        source_labels[:dev_rows],
        best["development"],
    )
    counts = [row["updates"] for row in epochs]
    checks = {
        "cohort": learning["cohort"] == HIGGS_CONFIRMATION_COHORT,
        "alpha": learning["alpha"] == 1 / learning["divisor"],
        "model_shapes": all(
            model_shapes.get(name) == shape for name, shape in expected_shapes.items()
        ),
        "parameter_inventory": learning["trainable_parameters"]
        == (4 * count if learning["kind"] == "cst" else width * width) + 31 * width + 2,
        "dataset_sha256": learning["dataset_sha256"] == data_sha,
        "initialization_sha256": file_sha256(directory / "initialization.pt")
        == learning["initialization_sha256"],
        "initial_configuration": all(
            initial["args"][k] == learning[k]
            for k in ("kind", "width", "divisor", "seed", "lr", "mode")
        ),
        "all_layers_trainable": learning["all_layers_trainable"] is True,
        "training_protocol": learning["batch_size"] == 4
        and learning["training_rows"] == 100_000
        and not learning["TF32"]
        and learning["cpu_math_threads"] == 1,
        "epoch_sequence": [r["epoch"] for r in epochs]
        == list(range(1, len(epochs) + 1)),
        "update_sequence": counts == [steps * n for n in range(1, len(epochs) + 1)]
        and learning["total_updates"] == counts[-1],
        "epoch_budget": len(epochs) == initial["args"]["epochs"],
        "step_budget": steps == (initial["args"]["steps_per_epoch"] or 25_000),
        "development_rows": dev_rows == DEV_ROWS
        or (
            dev_rows % 4 == 0
            and 4 <= dev_rows <= 32
            and steps <= 32
            and len(epochs) <= 2
        ),
        "development_partition": learning["validation_partition"]["development"]
        == [0, dev_rows],
        "confirmation_partition": learning["validation_partition"]["confirmation"]
        == [DEV_ROWS, DEV_ROWS + CONFIRMATION_ROWS],
        "no_confirmation_training": learning["confirmation_evaluated"] is False
        and learning["test_evaluated"] is False,
        "graph_replays": learning["graph_replays"]
        == (learning["total_updates"] - 8 if learning["mode"] == "graph" else 0),
        "selection_epoch": best["epoch"] == expected_best["epoch"],
        "selection_metrics": best["development"] == expected_best["development"],
        "selection_state": best["state"] == selected_row["state"]
        and best["gradients"] == selected_row["gradients"],
        "selection_manifest": best == learning["best_development_checkpoint"],
        "best_checkpoint_sha256": best["checkpoint_sha256"]
        == file_sha256(directory / "best-dev.pt"),
        "best_predictions_sha256": best["predictions_sha256"]
        == file_sha256(directory / "best-dev-predictions.npz"),
        "prediction_metrics_and_labels": predictions["pass"],
        "retained_checkpoints": all(row["pass"] for row in checkpoints),
    }
    result = {
        "directory": directory.name,
        "kind": learning["kind"],
        "width": learning["width"],
        "divisor": learning["divisor"],
        "seed": learning["seed"],
        "lr": learning["lr"],
        "mode": learning["mode"],
        "epochs": len(epochs),
        "steps_per_epoch": steps,
        "best_epoch": best["epoch"],
        "checkpoints": checkpoints,
        "development_predictions": predictions,
        "ordinary_initialization": {
            name: tensor_hash(initial["model"][name])
            for name in ("0.weight", "0.bias", "4.weight", "4.bias")
        },
        "dense_middle_initialization": tensor_hash(initial["model"]["2.weight"])
        if learning["kind"] == "dense"
        else None,
        "data_orders": {str(row["epoch"]): row["order_sha256"] for row in epochs},
        "checks": checks,
        "pass": all(checks.values()),
    }
    del initial
    gc.collect()
    return result


def verify_qualification_pairs(paths):
    import torch

    groups, comparisons = {}, []
    for path in paths:
        learning = read_json(path)
        rows = learning["epochs"]
        if len(rows) > 2 or rows[0]["updates"] > 32:
            continue
        key = tuple(learning[k] for k in ("kind", "width", "divisor", "seed", "lr"))
        entry = groups.setdefault(key, {})
        if learning["mode"] in entry:
            raise ValueError("duplicate qualification mode for one cohort")
        entry[learning["mode"]] = path.parent
    missing = []
    for key, entry in groups.items():
        if set(entry) != {"eager", "graph"}:
            missing.append(list(key))
            continue
        eager, graph = entry["eager"], entry["graph"]
        first = read_json(eager / "learning.json")
        second = read_json(graph / "learning.json")
        if len(first["epochs"]) != len(second["epochs"]):
            raise ValueError("qualification epoch budgets differ")
        for index, row in enumerate(first["epochs"], 1):
            paths_to_compare = [
                directory / f"epoch-{index}.pt" for directory in (eager, graph)
            ]
            left, right = [
                torch.load(path, map_location="cpu", weights_only=True)
                for path in paths_to_compare
            ]
            left_fields, right_fields = (
                dict(tensor_fields(left)),
                dict(tensor_fields(right)),
            )
            different = sorted(set(left_fields) ^ set(right_fields))
            different += [
                name
                for name in left_fields.keys() & right_fields.keys()
                if not byte_equal(left_fields[name], right_fields[name])
            ]
            same_updates = left["updates"] == right["updates"] == row["updates"]
            comparisons.append(
                {
                    "kind": key[0],
                    "width": key[1],
                    "divisor": key[2],
                    "seed": key[3],
                    "epoch": index,
                    "fields": len(left_fields),
                    "different_fields": different,
                    "same_updates": same_updates,
                    "pass": not different and same_updates,
                }
            )
            del left, right, left_fields, right_fields
            gc.collect()
    return {
        "comparisons": comparisons,
        "unpaired_cohorts": missing,
        "pass": not missing and all(row["pass"] for row in comparisons),
    }


def verify_evaluation(path, source_labels, data_sha, available_checkpoints):
    evaluation = read_json(path)
    predictions = verify_predictions(
        path.parent / "predictions.npz",
        source_labels[DEV_ROWS : DEV_ROWS + CONFIRMATION_ROWS],
        evaluation["confirmation"],
    )
    configuration = evaluation["checkpoint_configuration"]
    checks = {
        "cohort": evaluation["cohort"]
        == configuration["cohort"]
        == HIGGS_CONFIRMATION_COHORT,
        "identity": all(
            evaluation[k] == configuration[k]
            for k in ("kind", "width", "divisor", "seed", "lr", "dataset_sha256")
        ),
        "dataset_sha256": evaluation["dataset_sha256"] == data_sha,
        "checkpoint_retained": evaluation["checkpoint_sha256"] in available_checkpoints,
        "confirmation_partition": evaluation["validation_partition"]
        == [DEV_ROWS, DEV_ROWS + CONFIRMATION_ROWS],
        "prediction_sha256": evaluation["predictions_sha256"]
        == file_sha256(path.parent / "predictions.npz"),
        "logits_sha256": evaluation["logits_sha256"] == predictions["logits_sha256"],
        "labels_sha256": evaluation["labels_sha256"] == predictions["labels_sha256"],
        "metrics_and_labels": predictions["pass"],
        "no_test_evaluation": evaluation["test_evaluated"] is False,
    }
    return {
        "file": str(path),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "predictions": predictions,
        "checks": checks,
        "pass": all(checks.values()),
    }


def verify_oracle(path):
    import torch

    from benchmarks.training.owner_window_fp32 import errors

    values = torch.load(path, map_location="cpu", weights_only=True)
    if len(values["actual"]) != 3 or len(values["expected"]) != 3:
        raise ValueError("oracle requires retained Y/dX/all-dP")
    metrics = {
        name: errors(actual, expected)
        for name, actual, expected in zip(
            ("Y", "dX", "dP"), values["actual"], values["expected"]
        )
    }
    return {
        "file": path.name,
        "atoms": len(values["P"]),
        "metrics": metrics,
        "pass": all(value["pass"] for value in metrics.values()),
    }


def checkpoint_inventory(job, related_roots=()):
    """Hash actual checkpoint bytes, including recovery tar members, without extraction."""
    inventory = {}

    def record(stream, origin):
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        inventory.setdefault(digest.hexdigest(), []).append(origin)

    def archive_members(stream, origin, depth=0):
        if depth > 2:
            raise ValueError("unexpectedly nested recovery archives")
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith(".pt"):
                    with archive.extractfile(member) as payload:
                        record(payload, origin + "!" + name)
                elif name.endswith((".tar.gz", ".tgz", ".tar")):
                    with archive.extractfile(member) as payload:
                        archive_members(payload, origin + "!" + name, depth + 1)

    job = Path(job)
    for path in sorted((job / "results/artifacts").rglob("*.pt")):
        with path.open("rb") as stream:
            record(stream, str(path))
    archives = {job / "source.tar.gz"}
    archives.update((job / "results/artifacts").rglob("*.tar.gz"))
    for root in related_roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*.pt"))
            # Result/source archives include nested recovery state. Avoid a
            # second traversal of the same recovery archives when possible.
            candidates += sorted(root.rglob("source.tar.gz"))
            candidates += sorted(root.rglob("results.tar.gz"))
        for path in candidates:
            if path.suffix == ".pt":
                with path.open("rb") as stream:
                    record(stream, str(path))
            elif path.name.endswith((".tar.gz", ".tgz", ".tar")):
                archives.add(path)
    for path in sorted(archives):
        with path.open("rb") as stream:
            archive_members(stream, str(path))
    return inventory


def verify(job_directory, source_commit, repository=None, related_evidence_roots=()):
    import torch

    job = Path(job_directory)
    results, artifacts = job / "results", job / "results/artifacts"
    spec, receipt = read_json(job / "spec.json"), read_json(job / "receipt.json")
    manifest, execution = (
        read_json(results / "manifest.json"),
        read_json(results / "result.json"),
    )
    report = {
        "schema_version": 1,
        "job_id": spec["id"],
        "reader_torch": str(torch.__version__),
        "requested_source_commit": source_commit,
        "execution": execution,
        "sections": {},
        "failures": [],
    }

    def section(name, function):
        try:
            value = function()
        except Exception as error:  # noqa: BLE001 - preserve a partial audit
            value = {"pass": False, "error": type(error).__name__ + ": " + str(error)}
        report["sections"][name] = value
        if isinstance(value, dict) and value.get("pass") is False:
            report["failures"].append(name)
        return value

    section(
        "archive_sha256",
        lambda: {
            "pass": file_sha256(job / "source.tar.gz")
            == spec["source_sha256"]
            == execution["source_sha256"]
            and file_sha256(job / "results.tar.gz") == receipt["archive_sha256"]
        },
    )
    actual = {
        str(p.relative_to(results))
        for p in results.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    section(
        "result_manifest",
        lambda: {
            "files": len(manifest),
            "pass": actual == set(manifest)
            and all(
                file_sha256(safe_path(results, name)) == digest
                for name, digest in manifest.items()
            ),
        },
    )
    section(
        "source_archive_contents",
        lambda: verify_archive(job / "source.tar.gz", spec["source_files"]),
    )
    section(
        "result_archive_contents",
        lambda: verify_archive(
            job / "results.tar.gz",
            manifest | {"manifest.json": file_sha256(results / "manifest.json")},
        ),
    )
    section(
        "git_source",
        lambda: verify_git_source(
            repository or Path(__file__).resolve().parents[2],
            source_commit,
            spec["source_files"],
        ),
    )
    environment_path = artifacts / "environment.json"
    if environment_path.is_file():
        environment = read_json(environment_path)
        if "sources" in environment:
            section(
                "executed_python_sources",
                lambda: {
                    "files": len(environment["sources"]),
                    "pass": all(
                        spec["source_files"].get(name) == digest
                        for name, digest in environment["sources"].items()
                    ),
                },
            )
    source_name = "research-inputs/screen.npz"
    with tarfile.open(job / "source.tar.gz", "r:gz") as archive:
        stream = archive.extractfile(source_name)
        if stream is None:
            raise ValueError("source data payload missing")
        raw = stream.read()
    data_sha = hashlib.sha256(raw).hexdigest()
    if data_sha != spec["source_files"][source_name]:
        raise ValueError("source dataset bytes differ from snapshot")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        source_labels = data["y_val"].copy()
    del raw
    section(
        "source_validation_labels",
        lambda: {
            "rows": len(source_labels),
            "pass": len(source_labels) == DEV_ROWS + CONFIRMATION_ROWS
            and source_labels.dtype == np.int64
            and np.isin(source_labels, (0, 1)).all().item(),
        },
    )
    cells = []
    learning_paths = sorted(artifacts.rglob("learning.json"))
    for path in learning_paths:
        value = section(
            "cell/" + str(path.parent.relative_to(artifacts)),
            lambda path=path: verify_cell(path.parent, data_sha, source_labels),
        )
        if "ordinary_initialization" in value:
            cells.append(value)
    ordinary, dense, orders, mismatch = {}, {}, {}, []
    for cell in cells:
        key = (cell["width"], cell["seed"])
        if key in ordinary and ordinary[key] != cell["ordinary_initialization"]:
            mismatch.append("ordinary/" + cell["directory"])
        ordinary[key] = cell["ordinary_initialization"]
        if cell["kind"] == "dense":
            if key in dense and dense[key] != cell["dense_middle_initialization"]:
                mismatch.append("dense/" + cell["directory"])
            dense[key] = cell["dense_middle_initialization"]
        for epoch, digest in cell["data_orders"].items():
            key_order = (cell["seed"], epoch)
            if key_order in orders and orders[key_order] != digest:
                mismatch.append("order/" + cell["directory"] + "/" + epoch)
            orders[key_order] = digest
    section(
        "shared_initialization_orders",
        lambda: {"mismatches": mismatch, "pass": not mismatch},
    )
    section("qualification_pairs", lambda: verify_qualification_pairs(learning_paths))
    for path in sorted(artifacts.rglob("*.pt")):
        if path.name.endswith("-vjp.pt") or "quadrature" in path.name:
            section(
                "oracle/" + str(path.relative_to(artifacts)),
                lambda path=path: verify_oracle(path),
            )
    inventory = section(
        "checkpoint_inventory",
        lambda: {"checkpoints": checkpoint_inventory(job, related_evidence_roots)},
    )
    checkpoint_digests = set(inventory.get("checkpoints", {}))
    for path in sorted(artifacts.rglob("evaluation.json")):
        section(
            "evaluation/" + str(path.parent.relative_to(artifacts)),
            lambda path=path: verify_evaluation(
                path, source_labels, data_sha, checkpoint_digests
            ),
        )
    report["completed_training_cells"] = len(cells)
    report["evidence_verified"] = not report["failures"]
    report["execution_succeeded"] = (
        execution["returncode"] == 0 and not execution["timed_out"]
    )
    report["status"] = (
        "verified_evidence" if report["evidence_verified"] else "verification_failed"
    )
    report["scope"] = (
        "Archive, source, actual retained checkpoints, dev-only selection and prediction labels/metrics verified. No GPU replay, test quality or statistical noninferiority decision is performed here. Cross-phase freeze-manifest consistency must also be checked by the campaign analyst."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--related-evidence-root",
        action="append",
        type=Path,
        default=[],
        help="retained earlier job tree/archive; repeat to resolve cross-job checkpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"verification output already exists: {args.output}")
    report = verify(
        args.job_directory,
        args.source_commit,
        related_evidence_roots=args.related_evidence_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "job_id",
                    "status",
                    "completed_training_cells",
                    "evidence_verified",
                    "execution_succeeded",
                    "failures",
                )
            }
        )
    )
    if not report["evidence_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
