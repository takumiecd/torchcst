"""Staged HIGGS development tuning and frozen three-seed confirmation.

Only the deferred assess stage evaluates confirmation examples. Development
learning-rate selection, horizon selection and per-seed checkpoint selection
must be complete before confirmation-freeze.json is written. This module
never opens the official held-out test payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

WIDTHS = (1024, 2048)
DIVISORS = (1000, 500, 200, 100)
DEVELOPMENT_SEED = 17
CONFIRMATION_SEEDS = (29, 43, 71)
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003)
BASE_EPOCHS = 10
EXTENDED_EPOCHS = 20
LATE_IMPROVEMENT = 0.001


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"refusing to replace a frozen experiment record: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        destination.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def protocol():
    return {
        "schema_version": 1,
        "widths": list(WIDTHS),
        "divisors": list(DIVISORS),
        "alpha_family": [1 / divisor for divisor in DIVISORS],
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "base_learning_rates": list(LEARNING_RATES),
        "one_edge_extension_per_group": {"lower": 0.00003, "upper": 0.01},
        "base_epochs": BASE_EPOCHS,
        "extended_epochs": EXTENDED_EPOCHS,
        "horizon_trigger": (
            "any selected development config: best_epoch>=9 and "
            "epoch8 loss minus epoch10 loss>0.001"
        ),
        "extended_horizon_tuning": "rerun the top two development LRs per group for 20 epochs",
        "development_partition": [0, 50000],
        "confirmation_partition": [50000, 100000],
        "training_rows": 100000,
        "batch_size": 4,
        "checkpoint_selection": "minimum development CE; earliest exact tie",
        "learning_rate_selection": "minimum best-development CE; lower LR on exact tie",
        "accuracy_margin": 0.002,
        "roc_auc_margin": 0.002,
        "primary_statistics": "intersection-union of two one-sided paired-seed t tests; Bonferroni over 8 configurations",
        "primary_configuration_family_size": 8,
        "diagnostic_simultaneous_two_sided_endpoint_family_size": 16,
        "paired_example_bootstrap_replicates": 2000,
        "paired_example_bootstrap_seed": 20261105,
        "statistics": "preregistered higgs_noninferiority.py; no adaptive density additions",
        "test_evaluated": False,
    }


def _source_code_sha256():
    root = Path(__file__).resolve().parents[2]
    suffixes = {".py", ".cpp", ".cu", ".cuh", ".h", ".json"}
    files = [
        path
        for base in (root / "src/torchcst", root / "benchmarks/training")
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "__pycache__" not in path.parts
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def experiment_identity(data, source_identity):
    """Pure CPU source/data/protocol identity shared with the qualification gate."""
    code = _source_code_sha256()
    return {
        "source_identity": source_identity or code,
        "source_code_sha256": code,
        "dataset_sha256": file_sha256(data),
        "protocol_sha256": _json_sha256(protocol()),
    }


def configurations(width):
    if type(width) is not int or width not in WIDTHS:
        raise ValueError("choose an explicit confirmation width")
    return [{"id": "dense", "kind": "dense", "width": width, "divisor": 100}] + [
        {
            "id": f"cst-{divisor}",
            "kind": "cst",
            "width": width,
            "divisor": divisor,
        }
        for divisor in DIVISORS
    ]


def _configuration(width, kind, divisor):
    identifier = "dense" if kind == "dense" else f"cst-{divisor}"
    matches = [row for row in configurations(width) if row["id"] == identifier]
    if not matches or kind not in ("dense", "cst"):
        raise ValueError("choose a declared dense or CST configuration")
    return matches[0]


def select_candidate(candidates):
    """Predeclared CE selection with deterministic LR/epoch tie breaking."""
    if not candidates:
        raise ValueError("no completed development candidates")
    if len({candidate["lr"] for candidate in candidates}) != len(candidates):
        raise ValueError("learning rates must be unique within a tuning group")
    for candidate in candidates:
        if (
            not math.isfinite(candidate["best_development_loss"])
            or not math.isfinite(candidate["lr"])
            or candidate["lr"] <= 0
            or candidate["best_epoch"] < 1
        ):
            raise ValueError("invalid development candidate")
    return min(
        candidates,
        key=lambda candidate: (
            candidate["best_development_loss"],
            candidate["lr"],
            candidate["best_epoch"],
        ),
    )


def late_improvement(candidate, horizon):
    """Assess only the selected development trajectory, never confirmation data."""
    if horizon not in (BASE_EPOCHS, EXTENDED_EPOCHS):
        raise ValueError("undeclared learning horizon")
    curve = candidate["development_losses"]
    if len(curve) != horizon or any(not math.isfinite(value) for value in curve):
        raise ValueError("development trajectory does not match its budget")
    return bool(
        candidate["best_epoch"] >= horizon - 1
        and curve[-3] - curve[-1] > LATE_IMPROVEMENT
    )


def edge_extension(selected_lr):
    if selected_lr == min(LEARNING_RATES):
        return 0.00003
    if selected_lr == max(LEARNING_RATES):
        return 0.01
    return None


def _require_qualification(state, identity):
    record = _read_json(state / "qualification.json")
    if record.get("status") != "passed" or record.get("identity") != identity:
        raise ValueError("current source/data/protocol has not passed qualification")


def _require_identity(record, identity):
    if record.get("identity") != identity:
        raise ValueError(
            "persistent record belongs to a different source/data/protocol"
        )


def _copy_verified(source, destination, expected=None):
    actual = file_sha256(source)
    if expected is not None and actual != expected:
        raise ValueError(f"source artifact hash differs: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(destination) != actual:
        raise ValueError(f"copied artifact hash differs: {destination}")
    return actual


def _validate_cell(cell, config, seed, lr, epochs, identity):
    learning = _read_json(cell / "learning.json")
    best = _read_json(cell / "best-dev.json")
    expected = {
        "kind": config["kind"],
        "width": config["width"],
        "divisor": config["divisor"],
        "seed": seed,
        "lr": lr,
        "training_rows": 100000,
        "development_rows": 50000,
        "batch_size": 4,
        "dataset_sha256": identity["dataset_sha256"],
        "total_updates": epochs * 25000,
    }
    if any(learning.get(key) != value for key, value in expected.items()):
        raise ValueError(f"cell differs from declared identity/budget: {cell}")
    if len(learning["epochs"]) != epochs:
        raise ValueError("cell epoch inventory does not match its horizon")
    if learning.get("validation_partition") != {
        "development": [0, 50000],
        "confirmation": [50000, 100000],
    }:
        raise ValueError("cell validation partitions changed")
    curves = [row["development"]["loss"] for row in learning["epochs"]]
    if (
        not all(math.isfinite(value) for value in curves)
        or not 1 <= best["epoch"] <= epochs
        or best["development"]["loss"] != min(curves)
        or best["epoch"] != curves.index(min(curves)) + 1
        or best["development"]["examples"] != 50000
    ):
        raise ValueError("best checkpoint was not selected by development loss")
    checkpoint_sha = file_sha256(cell / "best-dev.pt")
    if checkpoint_sha != best["checkpoint_sha256"]:
        raise ValueError("best-development checkpoint hash differs")
    return {
        "lr": lr,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_development_loss": best["development"]["loss"],
        "best_development": best["development"],
        "development_losses": curves,
        "checkpoint_sha256": checkpoint_sha,
        "best_metadata_sha256": file_sha256(cell / "best-dev.json"),
        "learning_sha256": file_sha256(cell / "learning.json"),
        "cell_directory": str(cell),
        "training_seconds": learning["training_seconds"],
        "adoption_seconds": learning["adoption_seconds"],
    }


class Campaign:
    def __init__(self, args):
        self.args = args
        self.state = args.campaign_state
        self.identity = experiment_identity(args.data, args.source_identity)
        args.output.mkdir(parents=True, exist_ok=True)
        self.cells = args.output / "cells"
        self.cells.mkdir(exist_ok=False)
        self.record = {
            "schema_version": 1,
            "stage": args.stage,
            "status": "running",
            "identity": self.identity,
            "width": args.width,
            "kind": args.kind,
            "divisor": args.divisor,
            "seed": args.seed,
            "protocol": protocol(),
            "cells": [],
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        self.persist()

    def persist(self):
        _write_json(self.args.output / "phase.json", self.record)

    def cell(self, label, config, seed, lr, epochs):
        cell = self.cells / label
        command = [
            sys.executable,
            "-m",
            "benchmarks.training.higgs_confirmation",
            "--data",
            str(self.args.data.resolve()),
            "--output",
            str(cell.resolve()),
            "--kind",
            config["kind"],
            "--width",
            str(config["width"]),
            "--divisor",
            str(config["divisor"]),
            "--seed",
            str(seed),
            "--lr",
            str(lr),
            "--epochs",
            str(epochs),
        ]
        progress = {"label": label, "command": command, "status": "running"}
        self.record["cells"].append(progress)
        self.persist()
        print("START " + label, flush=True)
        with (self.cells / (label + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=14400,
            )
        candidate = _validate_cell(cell, config, seed, lr, epochs, self.identity)
        progress.update(status="completed", candidate=candidate)
        self.persist()
        print("DONE " + label, flush=True)
        return candidate

    def tune(self):
        config = _configuration(self.args.width, self.args.kind, self.args.divisor)
        selection_path = (
            self.state
            / "tuning"
            / str(config["width"])
            / config["id"]
            / "selection.json"
        )
        if selection_path.exists():
            selection = _read_json(selection_path)
            _require_identity(selection, self.identity)
            _write_json(self.args.output / "selection.json", selection)
            return {"reused": True, "selection": selection}
        if (self.state / "confirmation-freeze.json").exists():
            raise ValueError(
                "development tuning cannot change after confirmation freeze"
            )
        candidates = [
            self.cell(
                f"{config['id']}-lr-{lr:g}",
                config,
                DEVELOPMENT_SEED,
                lr,
                BASE_EPOCHS,
            )
            for lr in LEARNING_RATES
        ]
        extra = edge_extension(select_candidate(candidates)["lr"])
        if extra is not None:
            candidates.append(
                self.cell(
                    f"{config['id']}-edge-lr-{extra:g}",
                    config,
                    DEVELOPMENT_SEED,
                    extra,
                    BASE_EPOCHS,
                )
            )
        selected = select_candidate(candidates)
        selection = {
            "schema_version": 1,
            "identity": self.identity,
            "configuration": config,
            "development_seed": DEVELOPMENT_SEED,
            "horizon": BASE_EPOCHS,
            "base_learning_rates": list(LEARNING_RATES),
            "edge_extension": extra,
            "evaluated_learning_rates": sorted(row["lr"] for row in candidates),
            "selected_lr_on_evaluated_boundary": selected["lr"]
            in (
                min(row["lr"] for row in candidates),
                max(row["lr"] for row in candidates),
            ),
            "selected": selected,
            "candidates": candidates,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        _write_immutable(selection_path, selection)
        _write_json(self.args.output / "candidates.json", candidates)
        _write_json(self.args.output / "selection.json", selection)
        return {"reused": False, "selection": selection}

    def freeze_width(self):
        width = self.args.width
        frozen_path = self.state / "frozen" / f"{width}.json"
        if frozen_path.exists():
            record = _read_json(frozen_path)
            _require_identity(record, self.identity)
            _write_json(self.args.output / "frozen-width.json", record)
            return {"reused": True, "frozen": record}
        if (self.state / "confirmation-freeze.json").exists():
            raise ValueError("width settings cannot change after confirmation freeze")
        groups = []
        for config in configurations(width):
            path = self.state / "tuning" / str(width) / config["id"] / "selection.json"
            group = _read_json(path)
            _require_identity(group, self.identity)
            if group["configuration"] != config or group["horizon"] != BASE_EPOCHS:
                raise ValueError("tuning group differs from width/configuration")
            groups.append(group)
        triggers = [
            group["configuration"]["id"]
            for group in groups
            if late_improvement(group["selected"], BASE_EPOCHS)
        ]
        horizon = EXTENDED_EPOCHS if triggers else BASE_EPOCHS
        frozen_configs = []
        for group in groups:
            config = group["configuration"]
            if triggers:
                top_two = sorted(
                    group["candidates"],
                    key=lambda row: (
                        row["best_development_loss"],
                        row["lr"],
                        row["best_epoch"],
                    ),
                )[:2]
                candidates = [
                    self.cell(
                        f"{config['id']}-extended-lr-{row['lr']:g}",
                        config,
                        DEVELOPMENT_SEED,
                        row["lr"],
                        EXTENDED_EPOCHS,
                    )
                    for row in top_two
                ]
            else:
                candidates = group["candidates"]
            selected = select_candidate(candidates)
            frozen_configs.append(
                {
                    **config,
                    "lr": selected["lr"],
                    "horizon": horizon,
                    "selected_development": selected,
                    "horizon_candidates": candidates,
                    "base_tuning_candidates": group["candidates"],
                    "still_improving_at_horizon": late_improvement(selected, horizon),
                    "base_lr_boundary_flag": group["selected_lr_on_evaluated_boundary"],
                }
            )
        record = {
            "schema_version": 1,
            "identity": self.identity,
            "width": width,
            "horizon": horizon,
            "horizon_extension_trigger_configurations": triggers,
            "configurations": frozen_configs,
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "selection_uses_development_only": True,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        _write_immutable(frozen_path, record)
        _write_json(self.args.output / "frozen-width.json", record)
        return {"reused": False, "frozen": record}

    def _retain_confirmation(self, cell, destination, config, seed, selected):
        if destination.exists():
            raise FileExistsError("refusing to replace a retained confirmation model")
        pending = self.state / "pending" / uuid.uuid4().hex
        pending.mkdir(parents=True)
        files = {}
        for name in ("best-dev.pt", "best-dev.json", "learning.json"):
            files[name] = _copy_verified(cell / name, pending / name)
        if files["best-dev.pt"] != selected["checkpoint_sha256"]:
            raise ValueError("retained checkpoint differs from selected checkpoint")
        record = {
            "schema_version": 1,
            "identity": self.identity,
            "configuration": config,
            "seed": seed,
            "selected": selected,
            "files": files,
            "confirmation_evaluated": False,
            "test_evaluated": False,
        }
        _write_json(pending / "complete.json", record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(destination)
        return record

    def _retained(self, directory, config, seed):
        record = _read_json(directory / "complete.json")
        _require_identity(record, self.identity)
        if record["configuration"] != config or record["seed"] != seed:
            raise ValueError("retained confirmation configuration changed")
        for name, expected in record["files"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError("retained confirmation artifact changed")
        return record

    def confirm(self):
        width, seed = self.args.width, self.args.seed
        if seed not in CONFIRMATION_SEEDS:
            raise ValueError("choose one of the three confirmation seeds")
        frozen = _read_json(self.state / "frozen" / f"{width}.json")
        _require_identity(frozen, self.identity)
        if (self.state / "confirmation-freeze.json").exists():
            raise ValueError(
                "confirmation models cannot change after assessment freeze"
            )
        retained = []
        for selected in frozen["configurations"]:
            config = {key: selected[key] for key in ("id", "kind", "width", "divisor")}
            destination = (
                self.state / "confirmed" / str(width) / str(seed) / config["id"]
            )
            if destination.exists():
                row = self._retained(destination, config, seed)
            else:
                candidate = self.cell(
                    f"{config['id']}-seed-{seed}",
                    config,
                    seed,
                    selected["lr"],
                    frozen["horizon"],
                )
                row = self._retain_confirmation(
                    Path(candidate["cell_directory"]),
                    destination,
                    config,
                    seed,
                    candidate,
                )
            if (
                row["selected"]["lr"] != selected["lr"]
                or row["selected"]["epochs"] != frozen["horizon"]
            ):
                raise ValueError("confirmation model did not use frozen LR/horizon")
            retained.append(row)
        _write_json(self.args.output / "confirmed.json", retained)
        return {"confirmed": retained, "confirmation_evaluated": False}

    def assess(self):
        frozen_rows = []
        for width in WIDTHS:
            frozen = _read_json(self.state / "frozen" / f"{width}.json")
            _require_identity(frozen, self.identity)
            for seed in CONFIRMATION_SEEDS:
                for selected in frozen["configurations"]:
                    config = {
                        key: selected[key] for key in ("id", "kind", "width", "divisor")
                    }
                    directory = (
                        self.state / "confirmed" / str(width) / str(seed) / config["id"]
                    )
                    retained = self._retained(directory, config, seed)
                    if (
                        retained["selected"]["lr"] != selected["lr"]
                        or retained["selected"]["epochs"] != frozen["horizon"]
                    ):
                        raise ValueError(
                            "confirmation checkpoint differs from frozen hyperparameters"
                        )
                    frozen_rows.append(
                        {
                            "configuration": config,
                            "seed": seed,
                            "lr": selected["lr"],
                            "horizon": frozen["horizon"],
                            "best_epoch": retained["selected"]["best_epoch"],
                            "checkpoint_sha256": retained["files"]["best-dev.pt"],
                            "best_metadata_sha256": retained["files"]["best-dev.json"],
                            "learning_sha256": retained["files"]["learning.json"],
                            "relative_directory": str(
                                directory.relative_to(self.state)
                            ),
                        }
                    )
        if len(frozen_rows) != 30:
            raise ValueError("assessment requires all30 frozen confirmation models")
        freeze = {
            "schema_version": 1,
            "identity": self.identity,
            "configurations": frozen_rows,
            "configuration_count": 30,
            "confirmation_partition": [50000, 100000],
            "test_evaluated": False,
        }
        freeze_path = self.state / "confirmation-freeze.json"
        _write_immutable(freeze_path, freeze)
        _copy_verified(freeze_path, self.args.output / "confirmation-freeze.json")
        freeze_sha = file_sha256(freeze_path)
        # No evaluator is invoked until the complete immutable manifest above
        # exists and has been copied into this phase's retained artifacts.
        assessments = []
        for row in frozen_rows:
            config, seed = row["configuration"], row["seed"]
            checkpoint_dir = self.state / row["relative_directory"]
            persistent = (
                self.state
                / "assessed"
                / str(config["width"])
                / str(seed)
                / config["id"]
            )
            destination = (
                self.args.output
                / "evaluations"
                / str(config["width"])
                / str(seed)
                / config["id"]
            )
            if persistent.exists():
                saved = _read_json(persistent / "complete.json")
                if saved.get("confirmation_freeze_sha256") != freeze_sha:
                    raise ValueError("cached assessment belongs to a different freeze")
                destination.mkdir(parents=True)
                for name, expected in saved["files"].items():
                    _copy_verified(persistent / name, destination / name, expected)
            else:
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.training.higgs_confirmation",
                    "--data",
                    str(self.args.data.resolve()),
                    "--output",
                    str(checkpoint_dir.resolve()),
                    "--kind",
                    config["kind"],
                    "--width",
                    str(config["width"]),
                    "--divisor",
                    str(config["divisor"]),
                    "--seed",
                    str(seed),
                    "--evaluate-only-checkpoint",
                    str((checkpoint_dir / "best-dev.pt").resolve()),
                    "--evaluation-output",
                    str(destination.resolve()),
                ]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=7200,
                    )
                evaluation = _read_json(destination / "evaluation.json")
                if (
                    evaluation["checkpoint_sha256"] != row["checkpoint_sha256"]
                    or evaluation["dataset_sha256"] != self.identity["dataset_sha256"]
                    or evaluation["validation_partition"] != [50000, 100000]
                    or evaluation["checkpoint_epoch"] != row["best_epoch"]
                    or file_sha256(destination / "predictions.npz")
                    != evaluation["predictions_sha256"]
                ):
                    raise ValueError("assessment differs from frozen checkpoint/data")
                pending = self.state / "pending" / uuid.uuid4().hex
                pending.mkdir(parents=True)
                files = {
                    name: _copy_verified(destination / name, pending / name)
                    for name in ("evaluation.json", "predictions.npz")
                }
                saved = {
                    "schema_version": 1,
                    "identity": self.identity,
                    "confirmation_freeze_sha256": freeze_sha,
                    "configuration": config,
                    "seed": seed,
                    "files": files,
                }
                _write_json(pending / "complete.json", saved)
                persistent.parent.mkdir(parents=True, exist_ok=True)
                pending.rename(persistent)
            assessments.append(
                {
                    **row,
                    "evaluation_directory": str(
                        destination.relative_to(self.args.output)
                    ),
                    "confirmation_freeze_sha256": freeze_sha,
                    "evaluation_sha256": file_sha256(destination / "evaluation.json"),
                    "predictions_sha256": file_sha256(destination / "predictions.npz"),
                }
            )
            self.record["assessed_configurations"] = len(assessments)
            self.persist()
        index = {
            "schema_version": 1,
            "identity": self.identity,
            "confirmation_freeze_sha256": freeze_sha,
            "assessments": assessments,
            "confirmation_evaluated": True,
            "test_evaluated": False,
        }
        _write_json(self.args.output / "assess-index.json", index)
        self.record["confirmation_evaluated"] = True
        return index


def run(args):
    if args.manifest:
        print(json.dumps({"stage": args.stage, "protocol": protocol()}, indent=2))
        return
    if args.stage != "qualify" and args.width is None and args.stage != "assess":
        raise ValueError("this stage requires --width")
    if args.stage == "tune" and args.kind is None:
        raise ValueError("tune requires --kind")
    if args.stage == "qualify":
        from benchmarks.training.higgs_confirmation_qualification import run as qualify

        return qualify(args)
    campaign = Campaign(args)
    try:
        _require_qualification(campaign.state, campaign.identity)
        result = {
            "tune": campaign.tune,
            "freeze-width": campaign.freeze_width,
            "confirm": campaign.confirm,
            "assess": campaign.assess,
        }[args.stage]()
        campaign.record.update(status="completed", result=result)
        campaign.persist()
        return result
    except BaseException as error:
        campaign.record.update(
            status="failed", error=f"{type(error).__name__}: {error}"
        )
        campaign.persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        "--phase",
        dest="stage",
        choices=("qualify", "tune", "freeze-width", "confirm", "assess"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument(
        "--source-identity", default=os.environ.get("CST_CONFIRM_SOURCE_SHA256")
    )
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--kind", choices=("dense", "cst"))
    parser.add_argument("--divisor", type=int, choices=DIVISORS, default=100)
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="print the fixed CPU-only stage protocol without running models",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
