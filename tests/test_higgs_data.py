"""Data/split invariants for the HIGGS density research payload."""

import gzip
import hashlib
import io
import json

import numpy as np
import pytest

from benchmarks.training.higgs_data import array_metadata, prepare


def _source(tmp_path, rows=None, text=None):
    path = tmp_path / "higgs.csv.gz"
    if text is None:
        output = io.StringIO()
        np.savetxt(output, rows, delimiter=",", fmt="%.9g")
        text = output.getvalue()
    with gzip.open(path, "wt", encoding="ascii") as destination:
        destination.write(text)
    return path


def _rows():
    x = np.arange(9 * 28, dtype=np.float32).reshape(9, 28) / 8
    x[5:] += 10_000  # Would visibly change the moments if val/test leaked.
    x[:, -1] = 7
    y = np.arange(9) % 2
    return np.column_stack((y, x))


def _prepare(path, output, **options):
    return prepare(
        path,
        output,
        train_end=5,
        val_end=7,
        total_rows=9,
        screen_rows=3,
        retain_train_rows=4,
        chunk_rows=2,
        **options,
    )


def test_official_style_split_and_all_training_only_statistics(tmp_path):
    rows = _rows()
    path = _source(tmp_path, rows)
    output = tmp_path / "prepared"
    record = _prepare(path, output)
    raw = rows[:, 1:].astype(np.float32)
    mean = raw[:5].astype(np.float64).mean(axis=0)
    std = np.maximum(raw[:5].astype(np.float64).std(axis=0), 1e-6)
    expected = (raw - mean.astype(np.float32)) / std.astype(np.float32)
    assert mean[0] != raw[:3].mean(axis=0)[0]
    np.testing.assert_array_equal(record["normalization"]["train_mean_float64"], mean)
    np.testing.assert_allclose(
        record["normalization"]["train_std_float64"], std, rtol=1e-14
    )
    assert record["ranges"] == {
        "statistics_training": [0, 5],
        "screen_training": [0, 3],
        "retained_training": [0, 4],
        "validation": [5, 7],
        "heldout_test": [7, 9],
    }
    with np.load(output / "screen.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "x_train",
            "y_train",
            "x_val",
            "y_val",
            "mean",
            "std",
        }
        np.testing.assert_array_equal(data["x_train"], expected[:3])
        np.testing.assert_array_equal(data["x_val"], expected[5:7])
        np.testing.assert_array_equal(data["y_train"], rows[:3, 0].astype(np.int64))
        assert data["x_train"].dtype == np.float32
        assert data["y_train"].dtype == np.int64
        assert data["std"][-1] == np.float32(1e-6)
    with np.load(output / "heldout-test.npz", allow_pickle=False) as data:
        assert set(data.files) == {"x_test", "y_test", "mean", "std"}
        np.testing.assert_array_equal(data["x_test"], expected[7:])
        np.testing.assert_array_equal(data["y_test"], rows[7:, 0].astype(np.int64))
    with np.load(output / "retained-train.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x_train_raw"], raw[:4])
        np.testing.assert_array_equal(data["y_train"], rows[:4, 0].astype(np.int64))
    assert (
        record["source_compressed_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert json.loads((output / "manifest.json").read_text()) == record


def test_hashes_and_payloads_are_deterministic_and_cover_raw_and_normalized(tmp_path):
    rows = _rows()
    path = _source(tmp_path, rows)
    first = _prepare(path, tmp_path / "one")
    second = _prepare(path, tmp_path / "two")
    assert first == second
    for name, array_name, expected in (
        ("raw_arrays", "x_train", rows[:3, 1:].astype(np.float32)),
        ("raw_arrays", "y_train", rows[:3, 0].astype(np.uint8)),
    ):
        assert first[name][array_name] == array_metadata(expected)
    with np.load(tmp_path / "one/screen.npz", allow_pickle=False) as data:
        for name in data.files:
            assert first["normalized_arrays"][name] == array_metadata(data[name])


@pytest.mark.parametrize("kind", ("label", "nan", "columns", "blank", "short", "long"))
def test_malformed_rows_and_incomplete_source_are_rejected_without_output(
    tmp_path, kind
):
    rows = _rows()
    if kind == "label":
        rows[6, 0] = 2
    elif kind == "nan":
        rows[7, 5] = np.nan
    elif kind == "columns":
        rows = rows[:, :-1]
    elif kind == "short":
        rows = rows[:-1]
    elif kind == "long":
        rows = np.concatenate((rows, rows[:1]))
    if kind == "blank":
        output = io.StringIO()
        np.savetxt(output, rows, delimiter=",", fmt="%.9g")
        lines = output.getvalue().splitlines(keepends=True)
        lines[4] = "\n"
        path = _source(tmp_path, text="".join(lines))
    else:
        path = _source(tmp_path, rows)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        _prepare(path, output)
    assert not output.exists()


def test_existing_output_directory_is_never_overwritten(tmp_path):
    path = _source(tmp_path, _rows())
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "unrelated"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        _prepare(path, output)
    assert sentinel.read_text() == "preserve"


def test_no_retained_training_archive_when_disabled(tmp_path):
    path = _source(tmp_path, _rows())
    output = tmp_path / "prepared"
    record = prepare(
        path,
        output,
        train_end=5,
        val_end=7,
        total_rows=9,
        screen_rows=3,
        retain_train_rows=0,
        chunk_rows=2,
    )
    assert not (output / "retained-train.npz").exists()
    assert "retained_training_raw_arrays" not in record
