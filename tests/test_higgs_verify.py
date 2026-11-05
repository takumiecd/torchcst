"""Post-retrieval checks must reject changed evidence, not just read metrics."""

import hashlib
import io
import json
import subprocess
import tarfile

import pytest
import torch

from benchmarks.training.higgs_verify import (
    byte_equal,
    expected_cells,
    file_sha256,
    safe_path,
    tensor_fields,
    tensor_hash,
    verify_archive,
    verify_cell,
    verify_git_source,
)


def test_tensor_paths_hashes_and_exact_byte_comparison():
    value = torch.tensor([0.0, -0.0], dtype=torch.float64)
    fields = dict(
        tensor_fields({"model": {"weight": value}, "optimizer": [torch.tensor(8.0)]})
    )
    assert set(fields) == {"/model/weight", "/optimizer/0"}
    assert tensor_hash(value) == {
        "dtype": "torch.float64",
        "shape": [2],
        "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
    }
    assert byte_equal(value, value.clone())
    assert not byte_equal(value, value.abs())
    assert not byte_equal(value, value.float())
    assert tensor_hash(torch.ones(3, dtype=torch.bfloat16))["shape"] == [3]


@pytest.fixture
def cell(tmp_path):
    directory = tmp_path / "grid-1024-dense"
    directory.mkdir()
    initial_model = {
        "0.weight": torch.zeros(1024, 28),
        "0.bias": torch.zeros(1024),
        "2.weight": torch.zeros(1024, 1024),
        "4.weight": torch.zeros(2, 1024),
        "4.bias": torch.zeros(2),
    }
    # Synthetic retained-state fixture, not a claim that these updates ran.
    model = {name: value + 0.01 for name, value in initial_model.items()}
    names = ("0.weight", "0.bias", "4.weight", "4.bias", "2.weight")
    optimizer = {
        "state": {
            index: {
                "step": torch.tensor(25000.0),
                "exp_avg": torch.ones_like(model[name]),
                "exp_avg_sq": torch.ones_like(model[name]),
            }
            for index, name in enumerate(names)
        },
        "param_groups": [{"params": [0, 1, 2, 3]}, {"params": [4]}],
    }
    gradients = {name: torch.full_like(value, 0.125) for name, value in model.items()}
    arguments = {
        "kind": "dense",
        "width": 1024,
        "divisor": 20,
        "seed": 17,
        "lr": 0.001,
        "mode": "graph",
    }
    torch.save(
        {"model": initial_model, "args": arguments}, directory / "initialization.pt"
    )
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "gradients": gradients,
        "updates": 25000,
    }
    torch.save(checkpoint, directory / "epoch-1.pt")
    order = torch.randperm(100000, generator=torch.Generator().manual_seed(117))
    learning = arguments | {
        "initialization_sha256": file_sha256(directory / "initialization.pt"),
        "dataset_sha256": "data-digest",
        "batch_size": 4,
        "all_layers_trainable": True,
        "training_rows": 100000,
        "validation_rows": 8192,
        "TF32": False,
        "cpu_math_threads": 1,
        "torch": str(torch.__version__),
        "total_updates": 25000,
        "graph_replays": 24992,
        "epochs": [
            {
                "epoch": 1,
                "updates": 25000,
                "state": {
                    key: tensor_hash(value)
                    for key, value in tensor_fields(
                        {"model": model, "optimizer": optimizer}
                    )
                },
                "gradients": {
                    name: tensor_hash(value) for name, value in gradients.items()
                },
                "order_sha256": tensor_hash(order)["sha256"],
            }
        ],
    }
    (directory / "learning.json").write_text(json.dumps(learning))
    return directory


def test_checkpoint_verification_uses_retained_state_gradients_and_order(cell):
    report = verify_cell(cell, "data-digest")
    assert report["pass"], report["checks"]
    assert report["state_tensor_fields"] == 20
    assert report["gradient_tensor_fields"] == 5
    assert all(report["ordinary_parameter_changed"].values())
    assert all(clock == 25000 for clock in report["optimizer_clocks"].values())


def test_other_torch_version_order_recomputation_is_diagnostic_only(cell):
    path = cell / "learning.json"
    learning = json.loads(path.read_text())
    learning["torch"] = "2.0.0+cu000"
    learning["epochs"][0]["order_sha256"] = "recorded-other-runtime-order"
    path.write_text(json.dumps(learning))
    report = verify_cell(cell, "data-digest")
    assert report["pass"]
    assert "epoch_1_order_recomputed" not in report["checks"]
    assert report["recomputed_orders"]["1"]["matches_recorded"] is False
    assert report["order_hashes"]["1"] == "recorded-other-runtime-order"
    assert "diagnostic only" in report["order_recomputation_scope"]


@pytest.mark.parametrize(
    "change",
    ("model", "gradient", "clock", "frozen_input", "replays", "order", "budget"),
)
def test_checkpoint_or_published_metadata_tampering_is_detected(cell, change):
    checkpoint = torch.load(cell / "epoch-1.pt", map_location="cpu", weights_only=True)
    learning = json.loads((cell / "learning.json").read_text())
    expected = {
        "model": "model_optimizer_tensor_hashes",
        "gradient": "gradient_tensor_hashes",
        "clock": "optimizer_clocks",
        "frozen_input": "input_layer_changed",
        "replays": "replay_count",
        "order": "epoch_1_order_recomputed",
        "budget": "main_update_budget_25000",
    }[change]
    if change == "model":
        checkpoint["model"]["0.weight"][0, 0] += 1
    elif change == "gradient":
        checkpoint["gradients"]["0.weight"][0, 0] += 1
    elif change == "clock":
        checkpoint["optimizer"]["state"][0]["step"].sub_(1)
    elif change == "frozen_input":
        checkpoint["model"]["0.weight"].zero_()
        checkpoint["model"]["0.bias"].zero_()
    elif change == "replays":
        learning["graph_replays"] -= 1
    elif change == "order":
        learning["epochs"][0]["order_sha256"] = "changed"
    else:
        learning["total_updates"] -= 1
    torch.save(checkpoint, cell / "epoch-1.pt")
    (cell / "learning.json").write_text(json.dumps(learning))
    report = verify_cell(cell, "data-digest")
    assert not report["pass"]
    assert report["checks"][expected] is False


def test_finite_campaign_inventory_is_explicit():
    names = expected_cells()
    assert len(names) == 35
    assert len([name for name in names if name.startswith("grid-")]) == 21
    assert "grid-8192-cst-100" in names
    assert "grid-8192-cst-200" not in names


def test_archive_hashing_and_unsafe_path_rejection(tmp_path):
    archive = tmp_path / "files.tar.gz"
    with tarfile.open(archive, "w:gz") as destination:
        entry = tarfile.TarInfo("one.txt")
        entry.size = 3
        destination.addfile(entry, io.BytesIO(b"one"))
    digest = hashlib.sha256(b"one").hexdigest()
    assert verify_archive(archive, {"one.txt": digest})["pass"]
    assert not verify_archive(archive, {"one.txt": "changed"})["pass"]
    assert not verify_archive(archive, {"one.txt": digest, "missing.txt": digest})[
        "pass"
    ]
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")
    with pytest.raises(ValueError):
        safe_path(tmp_path, "/outside")


def test_git_blob_verification_rejects_wrong_commit_bytes_or_inventory(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Verifier Test",
            "-c",
            "user.email=verifier@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    files = {
        "source.py": file_sha256(source),
        "research-inputs/screen.npz": "payload",
        "__pool_driver__.py": "driver",
    }
    assert verify_git_source(tmp_path, commit, files)["pass"]
    assert not verify_git_source(tmp_path, commit, files | {"source.py": "changed"})[
        "pass"
    ]
    assert not verify_git_source(tmp_path, commit, {"other.py": "extra"})["pass"]
