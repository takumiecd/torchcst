"""Clock paths are a serialized schema, not a count of independent updates."""

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from benchmarks.training.graph_clock_schema import (
    CST_MIDDLE_CLOCK_PATH,
    CST_MODEL_CLOCK_PATH,
    OPTIMIZER_CLOCK_PATHS,
    ClockSchemaError,
    validate_clock_schema,
)


def clocks(kind="cst", value=8):
    result = {
        path: torch.tensor(float(value), dtype=torch.float32, device="cpu")
        for path in OPTIMIZER_CLOCK_PATHS
    }
    if kind == "cst":
        result[CST_MODEL_CLOCK_PATH] = result[CST_MIDDLE_CLOCK_PATH].clone()
    return result


@pytest.mark.parametrize("kind,count", [("cst", 6), ("dense", 5)])
@pytest.mark.parametrize("value", [0, 8, 264, 45000, 45256])
def test_exact_clock_schema_and_read_only_counts(kind, count, value):
    state = clocks(kind, value)
    before = {key: (tensor.clone(), tensor._version) for key, tensor in state.items()}
    result = validate_clock_schema(state, expected_clock=value, kind=kind)
    assert result["optimizer_clock_count"] == result["unique_optimizer_entry_count"] == 5
    assert result["serialized_clock_count"] == count
    assert result["clock_dtype"] == "torch.float32" and result["clock_shape"] == []
    assert result["clock_device"] == "cpu" and result["clock_layout"] == "torch.strided"
    assert set(result["serialized_clock_values"]) == set(state)
    assert not any(isinstance(value, torch.Tensor) for value in result.values())
    for key, (saved, version) in before.items():
        assert torch.equal(state[key], saved) and state[key]._version == version
    if kind == "cst":
        assert state[CST_MODEL_CLOCK_PATH] is not state[CST_MIDDLE_CLOCK_PATH]
        assert result["model_clock_alias"]["value_equal"]
        assert result["model_clock_alias"]["bytes_equal"]
    else:
        assert result["model_clock_alias"] is None


@pytest.mark.parametrize("kind", ["cst", "dense"])
@pytest.mark.parametrize("change", ["missing", "extra", "wrong_optimizer", "wrong_alias"])
def test_missing_extra_and_wrong_values_rejected(kind, change):
    state = clocks(kind)
    if change == "missing":
        del state[OPTIMIZER_CLOCK_PATHS[0]]
    elif change == "extra":
        state["/optimizer/state/5/step"] = torch.tensor(8.0, device="cpu")
    elif change == "wrong_optimizer":
        state[CST_MIDDLE_CLOCK_PATH] = torch.tensor(9.0, device="cpu")
    else:
        state[CST_MODEL_CLOCK_PATH] = torch.tensor(9.0, device="cpu")
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind=kind)


@pytest.mark.parametrize(
    "bad",
    [None, 8, 8.0, "8", torch.tensor([8.0]), torch.tensor(8),
     torch.tensor(8.0, dtype=torch.float64), torch.tensor(float("nan")),
     torch.tensor(float("inf")), torch.tensor(-8.0),
     torch.tensor(8.0, requires_grad=True), torch.nn.Parameter(torch.tensor(8.0)),
     torch.empty((), device="meta")],
)
def test_malformed_clock_entries_rejected(bad):
    state = clocks()
    state[OPTIMIZER_CLOCK_PATHS[0]] = bad
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind="cst")


@pytest.mark.parametrize("expected", [True, False, -1, 8.0, "8", None, 2**24 + 1, 2**200])
def test_malformed_expected_clock_rejected(expected):
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(clocks(), expected_clock=expected, kind="cst")


@pytest.mark.parametrize("state", [None, [], {0: torch.tensor(8.0)}])
def test_malformed_flat_mapping_rejected(state):
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind="cst")


@pytest.mark.parametrize("kind", [None, "CST", 1, True])
def test_malformed_kind_rejected(kind):
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(clocks(), expected_clock=8, kind=kind)


def test_negative_zero_does_not_match_frozen_positive_zero_bytes():
    state = clocks(value=0)
    state[OPTIMIZER_CLOCK_PATHS[0]] = torch.tensor(-0.0, device="cpu")
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=0, kind="cst")


def test_fractional_clock_cannot_pass_by_integer_truncation():
    state = clocks(value=8.5)
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind="cst")


def test_same_count_wrong_clock_path_is_rejected():
    state = clocks()
    state["/optimizer/state/5/step"] = state.pop(OPTIMIZER_CLOCK_PATHS[0])
    assert len(state) == 6
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind="cst")


def test_missing_cst_model_alias_path_is_rejected():
    state = clocks()
    del state[CST_MODEL_CLOCK_PATH]
    with pytest.raises(ClockSchemaError):
        validate_clock_schema(state, expected_clock=8, kind="cst")


def test_recorded_stage3_actual_evidence_clock_schema():
    path = Path(__file__).resolve().parents[1] / (
        "output/design-search-20261104/stage3-clock-schema-audit/"
        "formal-0-initial-cst-qualified.evidence.pt"
    )
    if not path.exists():
        pytest.skip("private preserved Stage3 CPU evidence not installed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    for phase, expected in (("start", 8), ("end", 264)):
        result = validate_clock_schema(saved[phase], expected_clock=expected, kind="cst")
        assert result["serialized_clock_count"] == 6
        assert result["optimizer_clock_count"] == 5


def test_frozen_stage1_stage2_recorded_clock_hashes_and_schemas():
    jobs = Path.home() / ".local/state/colab-l4-pool/jobs"
    stage1 = jobs / "l4job-b48fffb7e8894db7805dce763fc161fc/results/artifacts/cells"
    stage2 = jobs / "l4job-2bcc5ab92d02417ea71963385e73ecb4/results/artifacts"
    if not stage1.is_dir() or not stage2.is_dir():
        pytest.skip("private frozen Stage1/2 JSON evidence not installed")
    count = 0
    for stage, folder, pattern in (
        (1, stage1, "numerical-*.json"), (2, stage2, "lifecycle-*/learning.json")
    ):
        for path in sorted(folder.glob(pattern)):
            if path.name.endswith(".preflight.json"):
                continue
            row = json.loads(path.read_text())
            snapshots = (
                [(data["state"], row["clock_start"] + (0 if phase == "start" else int(phase[4:])))
                 for phase, data in row["checkpoints"].items()]
                if stage == 1 else
                [(data["state"], data["total_updates"]) for data in row["epochs"]]
            )
            for snapshot, expected in snapshots:
                step_hashes = {key: value for key, value in snapshot.items() if key.endswith("/step")}
                expected_digest = hashlib.sha256(struct.pack("<f", expected)).hexdigest()
                assert all(meta == {"dtype": "torch.float32", "shape": [], "sha256": expected_digest}
                           for meta in step_hashes.values())
                state = {key: torch.tensor(float(expected), dtype=torch.float32, device="cpu")
                         for key in step_hashes}
                result = validate_clock_schema(state, expected_clock=expected, kind=row["kind"])
                assert result["serialized_clock_count"] == (6 if row["kind"] == "cst" else 5)
                count += 1
    assert count == 56
