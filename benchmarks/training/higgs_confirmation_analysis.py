"""Verify the complete frozen HIGGS assessment, then compute paired inference.

Run on CPU with --assessment DIR --data screen.npz --output FRESH_DIR.
The official test split is never opened. Scores retain FP32 subtraction and
seed pairing; statistical definitions live in higgs_noninferiority.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from benchmarks.training.higgs_noninferiority import (
    CONFIRMATION_SEEDS,
    CONFIRMATION_WIDTHS,
    compare_predictions,
    minimum_alpha_summary,
)

DIVISORS = (1000, 500, 200, 100)
COHORT = "higgs_alpha_confirmation_v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def checked_path(root, relative):
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in ("..", ".") for p in path.parts)
    ):
        raise ValueError("unsafe assessment path")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("assessment paths must not contain symlinks")
    return current


def expected_cases():
    return {
        (width, seed, kind, divisor)
        for width in CONFIRMATION_WIDTHS
        for seed in CONFIRMATION_SEEDS
        for kind, divisor in [("dense", 100), *[("cst", d) for d in DIVISORS]]
    }


def case_key(row):
    config = row["configuration"]
    key = config["width"], row["seed"], config["kind"], config["divisor"]
    expected_id = "dense" if key[2] == "dense" else f"cst-{key[3]}"
    if key not in expected_cases() or config["id"] != expected_id:
        raise ValueError("undeclared configuration identity")
    return key


def indexed(rows):
    result = {}
    for row in rows:
        key = case_key(row)
        if key in result:
            raise ValueError("duplicate assessment identity")
        result[key] = row
    if set(result) != expected_cases():
        raise ValueError("assessment requires all 30 declared frozen models")
    return result


def verify_assessment(assessment, data):
    """Return verified FP32 scores keyed by width/seed/kind/divisor and labels."""
    assessment, data = Path(assessment).resolve(), Path(data).resolve()
    index = read_json(checked_path(assessment, "assess-index.json"))
    freeze_path = checked_path(assessment, "confirmation-freeze.json")
    freeze_sha = digest(freeze_path)
    freeze = read_json(freeze_path)
    identity = index["identity"]
    if (
        index["schema_version"] != 1
        or freeze["schema_version"] != 1
        or index["confirmation_evaluated"] is not True
        or index["test_evaluated"] is not False
        or freeze["test_evaluated"] is not False
        or freeze["configuration_count"] != 30
        or freeze["confirmation_partition"] != [50000, 100000]
        or identity != freeze["identity"]
        or identity["dataset_sha256"] != digest(data)
        or index["confirmation_freeze_sha256"] != freeze_sha
    ):
        raise ValueError("assessment/freeze/dataset identity mismatch")
    with np.load(data, allow_pickle=False) as source:
        validation = source["y_val"]
        if validation.dtype != np.int64 or validation.shape != (100000,):
            raise ValueError("expected the exact official 100000 validation labels")
        labels = validation[50000:100000].copy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("confirmation requires both binary classes")
    label_sha = hashlib.sha256(labels.tobytes()).hexdigest()
    frozen = indexed(freeze["configurations"])
    rows = indexed(index["assessments"])
    scores, evidence, settings = {}, [], {}
    runtime = None
    for key in sorted(rows):
        row, original = rows[key], frozen[key]
        if any(row.get(name) != value for name, value in original.items()):
            raise ValueError("assessment row differs from frozen selection")
        if row["confirmation_freeze_sha256"] != freeze_sha:
            raise ValueError("assessment row freeze checksum mismatch")
        width, seed, kind, divisor = key
        setting_key = width, kind, divisor
        setting = row["lr"], row["horizon"]
        if setting_key in settings and settings[setting_key] != setting:
            raise ValueError("learning rate/horizon differs across paired seeds")
        settings[setting_key] = setting
        if row["horizon"] not in (10, 20) or row["lr"] not in (
            0.0001,
            0.0003,
            0.001,
            0.003,
        ):
            raise ValueError("selection is outside the declared training grid")
        directory = checked_path(assessment, row["evaluation_directory"])
        evaluation_path = checked_path(directory, "evaluation.json")
        predictions_path = checked_path(directory, "predictions.npz")
        if digest(evaluation_path) != row["evaluation_sha256"]:
            raise ValueError("evaluation checksum mismatch")
        predictions_sha = digest(predictions_path)
        evaluation = read_json(evaluation_path)
        config = evaluation["checkpoint_configuration"]
        expected = {
            "cohort": COHORT,
            "kind": kind,
            "width": width,
            "divisor": divisor,
            "seed": seed,
            "lr": row["lr"],
            "dataset_sha256": identity["dataset_sha256"],
        }
        if (
            any(evaluation.get(name) != value for name, value in expected.items())
            or any(config.get(name) != value for name, value in expected.items())
            or config["epochs"] != row["horizon"]
            or evaluation["schema_version"] != 1
            or evaluation["alpha"] != 1 / divisor
            or evaluation["checkpoint_sha256"] != row["checkpoint_sha256"]
            or evaluation["checkpoint_epoch"] != row["best_epoch"]
            or not 1 <= row["best_epoch"] <= row["horizon"]
            or evaluation["checkpoint_updates"] != row["best_epoch"] * 25000
            or evaluation["validation_partition"] != [50000, 100000]
            or evaluation["confirmation"]["examples"] != 50000
            or evaluation["test_evaluated"] is not False
            or predictions_sha != row["predictions_sha256"]
            or predictions_sha != evaluation["predictions_sha256"]
        ):
            raise ValueError("evaluation does not match frozen model/data")
        current_runtime = {
            name: evaluation[name]
            for name in ("torch", "cuda", "gpu", "batch_size", "TF32")
        }
        if current_runtime["batch_size"] != 4 or current_runtime["TF32"] is not False:
            raise ValueError("evaluation execution protocol mismatch")
        if runtime is not None and runtime != current_runtime:
            raise ValueError("assessment runtimes differ")
        runtime = current_runtime
        with np.load(predictions_path, allow_pickle=False) as retained:
            if set(retained.files) != {"logits", "labels"}:
                raise ValueError("unexpected prediction archive inventory")
            logits, actual_labels = retained["logits"], retained["labels"]
        if (
            logits.dtype != np.float32
            or logits.shape != (50000, 2)
            or not np.isfinite(logits).all()
            or actual_labels.dtype != np.int64
            or not np.array_equal(actual_labels, labels)
            or evaluation["labels_sha256"] != label_sha
            or evaluation["logits_sha256"]
            != hashlib.sha256(logits.tobytes()).hexdigest()
        ):
            raise ValueError("retained logits/labels differ from confirmation identity")
        scores[key] = logits[:, 1] - logits[:, 0]
        evidence.append(
            {
                "width": width,
                "seed": seed,
                "kind": kind,
                "divisor": divisor,
                "evaluation_sha256": row["evaluation_sha256"],
                "predictions_sha256": predictions_sha,
            }
        )
    if any(
        len(
            {
                settings[(width, kind, divisor)][1]
                for kind, divisor in [("dense", 100), *[("cst", d) for d in DIVISORS]]
            }
        )
        != 1
        for width in CONFIRMATION_WIDTHS
    ):
        raise ValueError("all configurations at one width must use the same horizon")
    return (
        scores,
        labels,
        {
            "identity": identity,
            "confirmation_freeze_sha256": freeze_sha,
            "assess_index_sha256": digest(assessment / "assess-index.json"),
            "labels_sha256": label_sha,
            "runtime": runtime,
            "verified_cases": evidence,
        },
    )


def analyze(assessment, data):
    scores, labels, evidence = verify_assessment(assessment, data)
    comparisons = []
    for width in CONFIRMATION_WIDTHS:
        dense = np.stack(
            [scores[(width, seed, "dense", 100)] for seed in CONFIRMATION_SEEDS]
        )
        for divisor in DIVISORS:
            cst = np.stack(
                [scores[(width, seed, "cst", divisor)] for seed in CONFIRMATION_SEEDS]
            )
            comparisons.append(
                compare_predictions(cst, dense, labels, width=width, alpha=1 / divisor)
            )
    return {
        "schema_version": 1,
        **evidence,
        "confirmation_evaluated": True,
        "test_evaluated": False,
        "comparisons": comparisons,
        "minimum_alpha_summary": minimum_alpha_summary(comparisons),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"analysis output must be fresh: {args.output}")
    result = analyze(args.assessment, args.data)
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "comparisons.csv").open("w", newline="") as handle:
        columns = [
            "width",
            "alpha",
            "status",
            "primary_bonferroni_adjusted_p",
            "empirical_all_three_seed_margin_pass_candidate",
        ]
        columns += [
            f"{metric}_{suffix}"
            for metric in ("accuracy", "roc_auc")
            for suffix in ("mean_difference", "primary_one_sided_seed_t_lower")
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for comparison in result["comparisons"]:
            row = {name: comparison[name] for name in columns[:5]}
            row.update(
                {
                    f"{metric}_{suffix}": comparison["metrics"][metric][suffix]
                    for metric in ("accuracy", "roc_auc")
                    for suffix in ("mean_difference", "primary_one_sided_seed_t_lower")
                }
            )
            writer.writerow(row)
    return result


if __name__ == "__main__":
    main()
