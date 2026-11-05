"""Read-only local verification of retrieved HIGGS campaign evidence.

Archive/source integrity, retained checkpoint tensors, and numerical oracle
gates are checked independently of campaign completion. A failed or partial
campaign can therefore have verified evidence without being a completed screen.
Torch is imported only when tensor verification is reached; no GPU is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def file_sha256(path):
    with Path(path).open("rb") as source:
        return stream_sha256(source)


def stream_sha256(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe evidence path: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"evidence path leaves its directory: {name!r}")
    return path


def tensor_fields(value, prefix=""):
    """Same declared path convention as the measured whole-step snapshot."""
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_fields(item, prefix + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_fields(item, prefix + "/" + str(index))


def tensor_hash(value):
    import torch

    value = value.detach().cpu().contiguous()
    # Byte views also support bfloat16 and scalars without a large tobytes copy.
    raw = value.reshape(-1).view(torch.uint8).numpy()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(memoryview(raw)).hexdigest(),
    }


def finite_fields(fields):
    import torch

    return [
        name for name, value in fields.items() if not bool(torch.isfinite(value).all())
    ]


def byte_equal(left, right):
    import torch

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def verify_archive(path, expected):
    observed, bad, duplicates = set(), [], []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            name = member.name
            if (
                not member.isfile()
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
            ):
                raise ValueError(f"nonregular/unsafe archive entry: {name!r}")
            if name in observed:
                duplicates.append(name)
            observed.add(name)
            source = archive.extractfile(member)
            if source is None or stream_sha256(source) != expected.get(name):
                bad.append(name)
    return {
        "files": len(observed),
        "hash_mismatches": bad,
        "missing": sorted(set(expected) - observed),
        "unexpected": sorted(observed - set(expected)),
        "duplicates": duplicates,
        "pass": not (bad or duplicates or observed != set(expected)),
    }


def verify_git_source(repository, commit, source_files):
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise ValueError("source-commit must be a hexadecimal Git commit id")
    full_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", commit + "^{commit}"],
        cwd=repository,
        text=True,
    ).strip()
    tracked = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", full_commit],
            cwd=repository,
            text=True,
        ).splitlines()
    )
    excluded = sorted(
        name
        for name in source_files
        if name.startswith("research-inputs/") or name == "__pool_driver__.py"
    )
    supplied = set(source_files) - set(excluded)
    names = sorted(supplied & tracked)
    # cat-file returns the exact blob bytes that `git show COMMIT:path` uses,
    # without spawning a separate Git process for every tracked source file.
    requests = "".join(f"{full_commit}:{name}\n" for name in names).encode()
    blobs = subprocess.check_output(
        ["git", "cat-file", "--batch"], cwd=repository, input=requests
    )
    reader, mismatches = io.BytesIO(blobs), []
    for name in names:
        _identity, kind, length = reader.readline().split()
        if kind != b"blob":
            raise ValueError(f"nonblob Git entry: {name}")
        content = reader.read(int(length))
        if reader.read(1) != b"\n":
            raise ValueError("invalid git cat-file response framing")
        if hashlib.sha256(content).hexdigest() != source_files[name]:
            mismatches.append(name)
    return {
        "commit": full_commit,
        "tracked_files_verified": len(names),
        "excluded_payloads": excluded,
        "hash_mismatches": mismatches,
        "missing_tracked_files": sorted(tracked - supplied),
        "untracked_source_files": sorted(supplied - tracked),
        "pass": not (mismatches or tracked != supplied),
    }


def expected_cells():
    names = {
        f"qual-{divisor}-{kind}-{mode}"
        for divisor in (200, 10)
        for kind in ("cst", "dense")
        for mode in ("eager", "graph")
    }
    names.update(
        f"lr-{kind}-{lr}" for kind in ("cst", "dense") for lr in (0.0001, 0.001, 0.003)
    )
    for width in (1024, 2048, 4096, 8192):
        names.add(f"grid-{width}-dense")
        for divisor in (100, 20) if width == 8192 else (200, 100, 50, 20, 10):
            names.add(f"grid-{width}-cst-{divisor}")
    return names


def verify_cell(directory, data_sha256):
    import torch

    learning = read_json(directory / "learning.json")
    init_path = directory / "initialization.pt"
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    epochs = learning["epochs"]
    if not epochs:
        raise ValueError("learning.json has no completed epochs")
    last = epochs[-1]
    checkpoint_path = directory / f"epoch-{last['epoch']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = dict(
        tensor_fields(
            {"model": checkpoint["model"], "optimizer": checkpoint["optimizer"]}
        )
    )
    state_hashes = {name: tensor_hash(value) for name, value in state.items()}
    gradients = {
        name: tensor_hash(value) for name, value in checkpoint["gradients"].items()
    }
    clocks = {
        str(name): float(value["step"].item())
        for name, value in checkpoint["optimizer"]["state"].items()
    }
    initial_tensors = dict(tensor_fields(initial["model"]))
    finite = finite_fields(state) + [
        "gradients" + name
        for name in finite_fields(dict(tensor_fields(checkpoint["gradients"])))
    ]
    finite += ["initial" + name for name in finite_fields(initial_tensors)]
    ordinary = {
        name: tensor_hash(initial["model"][name])
        for name in ("0.weight", "0.bias", "4.weight", "4.bias")
    }
    changed = {
        name: not byte_equal(initial["model"][name], checkpoint["model"][name])
        for name in ordinary
    }
    updates = learning["total_updates"]
    checks = {
        "initialization_file_sha256": file_sha256(init_path)
        == learning["initialization_sha256"],
        "data_sha256": learning["dataset_sha256"] == data_sha256,
        "model_optimizer_tensor_hashes": state_hashes == last["state"],
        "gradient_tensor_hashes": gradients == last["gradients"],
        "all_retained_tensors_finite": not finite,
        "checkpoint_update_count": checkpoint["updates"] == last["updates"] == updates,
        "optimizer_clock_count": len(clocks) == 5,
        "optimizer_clocks": bool(clocks)
        and all(clock == updates for clock in clocks.values()),
        "input_layer_changed": changed["0.weight"] or changed["0.bias"],
        "output_layer_changed": changed["4.weight"] or changed["4.bias"],
        "all_layers_trainable_declared": learning["all_layers_trainable"] is True,
        "true_batch_four": learning["batch_size"] == 4,
        "training_row_budget_100000": learning["training_rows"] == 100000,
        "seed_17": learning["seed"] == 17,
        "fp32_ieee_cpu_threads_one": learning["TF32"] is False
        and learning["cpu_math_threads"] == 1,
        "replay_count": learning["graph_replays"]
        == (updates - 8 if learning["mode"] == "graph" else 0),
        "initial_configuration": all(
            initial["args"][name] == learning[name]
            for name in ("kind", "width", "divisor", "seed", "lr", "mode")
        ),
    }
    name = directory.name
    checks["declared_cell_name"] = name in expected_cells()
    if name.startswith("grid-"):
        checks["main_update_budget_25000"] = updates == 25000 and len(epochs) == 1
        checks["cell_identity"] = (
            name
            == (
                f"grid-{learning['width']}-{learning['kind']}"
                + (f"-{learning['divisor']}" if learning["kind"] == "cst" else "")
            )
            and learning["mode"] == "graph"
        )
    elif name.startswith("lr-"):
        checks["lr_update_budget_5000"] = updates == 5000 and len(epochs) == 1
        checks["cell_identity"] = (
            name == f"lr-{learning['kind']}-{learning['lr']}"
            and learning["width"] == 1024
            and learning["divisor"] == 20
            and learning["mode"] == "graph"
        )
    elif name.startswith("qual-"):
        checks["qualification_update_budget"] = len(epochs) == 2 and [
            row["updates"] for row in epochs
        ] == [16, 32]
        checks["cell_identity"] = (
            name == f"qual-{learning['divisor']}-{learning['kind']}-{learning['mode']}"
            and learning["width"] == 1024
        )
    checks["validation_row_budget"] = learning["validation_rows"] == (
        32 if name.startswith("qual-") else 8192
    )
    order_hashes, recomputed_orders = {}, {}
    reader_torch = str(torch.__version__)
    measured_torch = learning.get("torch")
    same_torch_version = (
        measured_torch is not None
        and reader_torch.split("+")[0] == measured_torch.split("+")[0]
    )
    for epoch in epochs:
        number = epoch["epoch"]
        order = torch.randperm(
            learning["training_rows"],
            generator=torch.Generator().manual_seed(
                learning["seed"] + 100 + number - 1
            ),
        )
        digest = tensor_hash(order)["sha256"]
        recomputed_orders[str(number)] = {
            "sha256": digest,
            "matches_recorded": digest == epoch["order_sha256"],
        }
        if same_torch_version:
            checks[f"epoch_{number}_order_recomputed"] = digest == epoch["order_sha256"]
        # Cross-case order identity always uses the measured runtime's hashes.
        # A changed reader RNG implementation must not mislabel correct remote
        # ordering as corrupt evidence.
        order_hashes[str(number)] = epoch["order_sha256"]
    groups = checkpoint["optimizer"]["param_groups"]
    owned = [identifier for group in groups for identifier in group["params"]]
    checks["optimizer_parameter_ownership"] = (
        len(groups) == 2
        and [len(group["params"]) for group in groups] == [4, 1]
        and len(owned) == len(set(owned)) == 5
        and set(owned) == set(checkpoint["optimizer"]["state"])
    )
    row = {
        "name": name,
        "kind": learning["kind"],
        "width": learning["width"],
        "divisor": learning["divisor"],
        "seed": learning["seed"],
        "lr": learning["lr"],
        "epochs": len(epochs),
        "updates": updates,
        "graph_replays": learning["graph_replays"],
        "retained_checkpoint": checkpoint_path.name,
        "state_tensor_fields": len(state),
        "gradient_tensor_fields": len(gradients),
        "optimizer_clocks": clocks,
        "ordinary_parameter_changed": changed,
        "nonfinite_fields": finite,
        "checks": checks,
        "pass": all(checks.values()),
        "initial_ordinary": ordinary,
        "order_hashes": order_hashes,
        "reader_torch": reader_torch,
        "measured_torch": measured_torch,
        "recomputed_orders": recomputed_orders,
        "order_recomputation_scope": "same Torch base version; hard gate"
        if same_torch_version
        else "different/unrecorded Torch base version; diagnostic only; recorded cross-case order hashes remain the identity gate",
        "initial_dense_middle": tensor_hash(initial["model"]["2.weight"])
        if learning["kind"] == "dense"
        else None,
        "logged_extension": learning.get("extension"),
    }
    del initial, checkpoint, state, initial_tensors
    gc.collect()
    return row


def verify_qualification_pairs(cells):
    import torch

    rows = []
    for divisor in (200, 10):
        for kind in ("cst", "dense"):
            for epoch in (1, 2):
                paths = [
                    cells / f"qual-{divisor}-{kind}-{mode}" / f"epoch-{epoch}.pt"
                    for mode in ("eager", "graph")
                ]
                if not all(path.is_file() for path in paths):
                    continue
                left, right = [
                    torch.load(path, map_location="cpu", weights_only=True)
                    for path in paths
                ]
                a, b = dict(tensor_fields(left)), dict(tensor_fields(right))
                different = sorted(set(a) ^ set(b))
                different += [
                    name
                    for name in a.keys() & b.keys()
                    if not byte_equal(a[name], b[name])
                ]
                nonfinite = finite_fields(a) + finite_fields(b)
                rows.append(
                    {
                        "divisor": divisor,
                        "kind": kind,
                        "epoch": epoch,
                        "tensor_fields": len(a),
                        "different_fields": different,
                        "nonfinite_fields": nonfinite,
                        "updates": [left["updates"], right["updates"]],
                        "pass": not (different or nonfinite)
                        and left["updates"] == right["updates"] == 16 * epoch,
                    }
                )
                del left, right, a, b
                gc.collect()
    return rows


def verify_oracles(artifacts):
    import torch

    from benchmarks.training.owner_window_fp32 import errors

    metadata = {row["name"]: row for row in read_json(artifacts / "oracles.json")}
    rows = []
    for name in ("full_min_density", "quadrature31"):
        path = artifacts / (name + ".pt")
        if not path.is_file():
            continue
        retained = torch.load(path, map_location="cpu", weights_only=True)
        recomputed = {
            key: errors(actual, expected)
            for key, actual, expected in zip(
                ("Y", "dX", "dP"), retained["actual"], retained["expected"]
            )
        }
        mismatches = []
        logged = metadata.get(name, {})
        for key, record in recomputed.items():
            for metric, value in record.items():
                prior = logged.get(key, {}).get(metric)
                equal = (
                    math.isclose(value, prior, rel_tol=1e-9, abs_tol=1e-12)
                    if type(value) is float and type(prior) in (float, int)
                    else value == prior
                )
                if not equal:
                    mismatches.append(key + "." + metric)
        finite = finite_fields(dict(tensor_fields(retained)))
        expected_atoms = 5242 if name == "full_min_density" else 31
        shape_ok = (
            tuple(retained["P"].shape) == (expected_atoms, 4)
            and tuple(retained["X"].shape) == tuple(retained["dY"].shape) == (4, 1024)
            and len(retained["actual"]) == len(retained["expected"]) == 3
            and [tuple(value.shape) for value in retained["actual"]]
            == [(4, 1024), (4, 1024), (expected_atoms, 4)]
            and [tuple(value.shape) for value in retained["expected"]]
            == [(4, 1024), (4, 1024), (expected_atoms, 4)]
        )
        rows.append(
            {
                "name": name,
                "atoms": expected_atoms,
                "recomputed_errors": recomputed,
                "logged_error_mismatches": mismatches,
                "nonfinite_fields": finite,
                "pass": shape_ok
                and not (mismatches or finite)
                and all(record["pass"] for record in recomputed.values()),
            }
        )
        del retained
    return rows


def verify(job_directory, source_commit, repository=None):
    job = Path(job_directory).resolve()
    repository = Path(repository) if repository else Path(__file__).resolve().parents[2]
    results, artifacts = job / "results", job / "results" / "artifacts"
    report = {
        "schema_version": 1,
        "job_directory": str(job),
        "requested_source_commit": source_commit,
        "failures": [],
        "sections": {},
        "scope": "CPU read-only verification of retained files/tensors and recorded numerical gates; no GPU execution or heldout test quality evaluation",
        "binary_verification_scope": "compiled CUDA extension SHA/path is logged per CST cell; binary bytes were not retained and cannot be independently rehashed locally",
    }

    def section(name, operation):
        try:
            value = operation()
            report["sections"][name] = value
            if isinstance(value, dict) and value.get("pass") is False:
                report["failures"].append(name)
            return value
        except Exception as error:  # noqa: BLE001 - preserve partial verification report
            report["sections"][name] = {
                "pass": False,
                "error": type(error).__name__ + ": " + str(error),
            }
            report["failures"].append(name)
            return None

    spec = read_json(job / "spec.json")
    receipt = read_json(job / "receipt.json")
    result = read_json(results / "result.json")
    manifest = read_json(results / "manifest.json")
    campaign = (
        read_json(artifacts / "campaign.json")
        if (artifacts / "campaign.json").is_file()
        else {}
    )
    report["job_id"] = spec["id"]
    report["execution"] = result
    report["campaign_status"] = campaign.get("status", "unavailable")
    report["campaign_error"] = campaign.get("error")

    def archive_hashes():
        source_digest = file_sha256(job / "source.tar.gz")
        result_digest = file_sha256(job / "results.tar.gz")
        return {
            "source_sha256": source_digest,
            "result_sha256": result_digest,
            "pass": source_digest == spec["source_sha256"] == result["source_sha256"]
            and result_digest == receipt["archive_sha256"],
        }

    section("archive_sha256", archive_hashes)

    def result_files():
        mismatches = [
            name
            for name, digest in manifest.items()
            if file_sha256(safe_path(results, name)) != digest
        ]
        actual = {
            str(path.relative_to(results))
            for path in results.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        return {
            "files": len(manifest),
            "hash_mismatches": mismatches,
            "unmanifested_files": sorted(actual - set(manifest)),
            "pass": not mismatches and actual == set(manifest),
        }

    section("result_manifest", result_files)
    section(
        "result_archive_contents",
        lambda: verify_archive(
            job / "results.tar.gz",
            manifest | {"manifest.json": file_sha256(results / "manifest.json")},
        ),
    )
    section(
        "source_archive_contents",
        lambda: verify_archive(job / "source.tar.gz", spec["source_files"]),
    )
    section(
        "git_source",
        lambda: verify_git_source(repository, source_commit, spec["source_files"]),
    )
    environment = (
        read_json(artifacts / "environment.json")
        if (artifacts / "environment.json").is_file()
        else {}
    )
    if environment:
        section(
            "executed_python_sources",
            lambda: {
                "files": len(environment["sources"]),
                "mismatches": [
                    name
                    for name, digest in environment["sources"].items()
                    if spec["source_files"].get(name) != digest
                ],
                "pass": all(
                    spec["source_files"].get(name) == digest
                    for name, digest in environment["sources"].items()
                ),
            },
        )
    data_sha256 = environment.get(
        "data_sha256", spec["source_files"].get("research-inputs/screen.npz")
    )
    section(
        "dataset_identity",
        lambda: {
            "sha256": data_sha256,
            "pass": data_sha256 is not None
            and data_sha256 == spec["source_files"].get("research-inputs/screen.npz"),
        },
    )
    cells = artifacts / "cells"
    paths = sorted(cells.glob("*/learning.json"))
    case_rows = []
    for path in paths:
        value = section(
            "cell/" + path.parent.name,
            lambda path=path: verify_cell(path.parent, data_sha256),
        )
        if value is not None:
            case_rows.append(value)
    report["completed_cells"] = [path.parent.name for path in paths]
    report["partial_cell_directories"] = (
        sorted(
            path.name
            for path in cells.iterdir()
            if path.is_dir() and not (path / "learning.json").is_file()
        )
        if cells.is_dir()
        else []
    )
    report["partial_checkpoint_numeric_scope"] = (
        "partial cells without learning.json have manifest integrity verification only"
    )

    def shared_initializations():
        ordinary, dense, orders, wrong = {}, {}, {}, []
        for row in case_rows:
            key = row["width"], row["seed"]
            value = row["initial_ordinary"]
            if key in ordinary and ordinary[key] != value:
                wrong.append(row["name"] + ":ordinary")
            ordinary[key] = value
            if row["kind"] == "dense":
                value = row["initial_dense_middle"]
                if key in dense and dense[key] != value:
                    wrong.append(row["name"] + ":dense_middle")
                dense[key] = value
            for epoch, digest in row["order_hashes"].items():
                key_order = row["seed"], epoch
                if key_order in orders and orders[key_order] != digest:
                    wrong.append(row["name"] + ":order_epoch_" + epoch)
                orders[key_order] = digest
        return {
            "ordinary_groups": len(ordinary),
            "dense_groups": len(dense),
            "order_groups": len(orders),
            "mismatches": wrong,
            "pass": not wrong,
        }

    section("shared_initialization_and_orders", shared_initializations)
    pairs = section(
        "qualification_byte_comparisons", lambda: verify_qualification_pairs(cells)
    )
    if pairs is not None and any(not row["pass"] for row in pairs):
        report["failures"].append("qualification_byte_comparisons")
    if (artifacts / "oracles.json").is_file():
        oracles = section("oracle_gates", lambda: verify_oracles(artifacts))
        if oracles is not None and any(not row["pass"] for row in oracles):
            report["failures"].append("oracle_gates")
    else:
        oracles = []
    recorded = {row["name"] for row in campaign.get("completed", [])}
    actual, expected = set(report["completed_cells"]), expected_cells()
    report["inventory"] = {
        "expected": sorted(expected),
        "missing": sorted(expected - actual),
        "unrecorded_completed": sorted(actual - recorded),
        "recorded_missing": sorted(recorded - actual),
    }
    if recorded - actual:
        report["failures"].append("recorded_completed_cell_missing")
    for row in case_rows:
        if row["name"].startswith("grid-") and row["lr"] != campaign.get(
            "selected_lr", {}
        ).get(row["kind"]):
            report["failures"].append("selected_learning_rate/" + row["name"])
    complete = campaign.get("status") == "complete_screen_only"
    complete_evidence = (
        actual == expected
        and pairs is not None
        and len(pairs) == 8
        and oracles is not None
        and len(oracles) == 2
    )
    if complete and (
        not complete_evidence or result["returncode"] != 0 or result["timed_out"]
    ):
        report["failures"].append("inconsistent_complete_campaign")
    report["screen_complete"] = (
        complete and complete_evidence and not report["failures"]
    )
    report["evidence_verified"] = not report["failures"]
    report["status"] = (
        "verification_failed"
        if report["failures"]
        else "verified_complete_screen"
        if report["screen_complete"]
        else "verified_incomplete_screen"
    )
    try:
        import torch

        report["reader_torch"] = str(torch.__version__)
    except ImportError:
        report["reader_torch"] = None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.job_directory, args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "job_id",
                    "status",
                    "evidence_verified",
                    "screen_complete",
                    "completed_cells",
                    "failures",
                )
            },
            indent=2,
        )
    )
    if not report["evidence_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
