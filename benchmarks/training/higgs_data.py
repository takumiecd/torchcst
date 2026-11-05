"""Prepare bounded HIGGS screening payloads from the official complete CSV.

This benchmark utility reads every official row so the tail validation/test
splits and training-only normalization remain identical across screening sizes.
The GPU screening archive deliberately contains no held-out test examples.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

FEATURES = 28
DEFAULT_TRAIN_END = 10_400_000
DEFAULT_VAL_END = 10_500_000
DEFAULT_TOTAL_ROWS = 11_000_000


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_metadata(value):
    """Hash stable C-order array bytes, recording their interpretation."""
    value = np.ascontiguousarray(value)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive Python integer")


def prepare(
    csv_path,
    outdir,
    *,
    train_end=DEFAULT_TRAIN_END,
    val_end=DEFAULT_VAL_END,
    total_rows=DEFAULT_TOTAL_ROWS,
    screen_rows=100_000,
    retain_train_rows=1_000_000,
    chunk_rows=8192,
):
    """Read a complete CSV(.gz), save separate screening/test/raw payloads.

    Row ranges are zero-based half-open. Population mean/std are accumulated
    over *all* ``[0, train_end)`` rows using FP64 chunk sums. The std floor is
    1e-6. Means/stds are then cast once to FP32 and each input is normalized as
    ``(raw_fp32 - mean_fp32) / std_fp32`` with FP32 arithmetic. Labels stored in
    archives are int64. Raw-array provenance hashes use source FP32 features and
    uint8 labels. No existing output directory or files are overwritten.
    """
    csv_path, outdir = Path(csv_path), Path(outdir)
    for name, value in (
        ("train_end", train_end),
        ("val_end", val_end),
        ("total_rows", total_rows),
        ("screen_rows", screen_rows),
        ("chunk_rows", chunk_rows),
    ):
        _positive_integer(name, value)
    if type(retain_train_rows) is not int or retain_train_rows < 0:
        raise ValueError("retain_train_rows must be a nonnegative Python integer")
    if not 0 < train_end < val_end < total_rows:
        raise ValueError("splits require 0 < train_end < val_end < total_rows")
    if screen_rows > train_end or retain_train_rows > train_end:
        raise ValueError("screen/retained training rows cannot exceed train_end")
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite HIGGS output: {outdir}")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    retained = max(screen_rows, retain_train_rows)
    train_x = np.empty((retained, FEATURES), dtype="<f4")
    train_y = np.empty(retained, dtype=np.uint8)
    val_x = np.empty((val_end - train_end, FEATURES), dtype="<f4")
    val_y = np.empty(val_end - train_end, dtype=np.uint8)
    test_x = np.empty((total_rows - val_end, FEATURES), dtype="<f4")
    test_y = np.empty(total_rows - val_end, dtype=np.uint8)
    sums = np.zeros(FEATURES, dtype=np.float64)
    squares = np.zeros(FEATURES, dtype=np.float64)
    count = 0
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt", encoding="ascii") as source:
        while lines := list(itertools.islice(source, chunk_rows)):
            start, end = count, count + len(lines)
            if end > total_rows:
                raise ValueError(f"HIGGS has more than {total_rows} rows")
            try:
                rows = np.loadtxt(lines, delimiter=",", dtype="<f4", ndmin=2)
            except ValueError as error:
                raise ValueError(
                    f"invalid HIGGS CSV block at rows [{start}, {end})"
                ) from error
            if (
                rows.shape != (len(lines), FEATURES + 1)
                or not np.isfinite(rows).all()
                or not np.isin(rows[:, 0], (0, 1)).all()
            ):
                raise ValueError(
                    f"invalid HIGGS rows [{start}, {end}): expected binary label "
                    f"and {FEATURES} finite features per row"
                )
            x, y = rows[:, 1:], rows[:, 0].astype(np.uint8)
            stop = min(end, train_end)
            if start < stop:
                block = x[: stop - start].astype(np.float64)
                sums += block.sum(axis=0)
                squares += np.square(block).sum(axis=0)
            for destination_x, destination_y, range_start, range_end in (
                (train_x, train_y, 0, retained),
                (val_x, val_y, train_end, val_end),
                (test_x, test_y, val_end, total_rows),
            ):
                first, last = max(start, range_start), min(end, range_end)
                if first < last:
                    target = slice(first - range_start, last - range_start)
                    selected = slice(first - start, last - start)
                    destination_x[target] = x[selected]
                    destination_y[target] = y[selected]
            count = end
    if count != total_rows:
        raise ValueError(f"expected {total_rows} HIGGS rows, found {count}")

    mean64 = sums / train_end
    std64 = np.maximum(
        np.sqrt(np.maximum(squares / train_end - mean64 * mean64, 0)), 1e-6
    )
    mean, std = mean64.astype("<f4"), std64.astype("<f4")
    screen_x = (train_x[:screen_rows] - mean) / std
    screen_y = train_y[:screen_rows].astype("<i8")
    normalized_val_x = (val_x - mean) / std
    normalized_test_x = (test_x - mean) / std
    normalized_val_y, normalized_test_y = val_y.astype("<i8"), test_y.astype("<i8")
    raw_arrays = {
        "x_train": train_x[:screen_rows],
        "y_train": train_y[:screen_rows],
        "x_val": val_x,
        "y_val": val_y,
        "x_test": test_x,
        "y_test": test_y,
    }
    normalized_arrays = {
        "x_train": screen_x,
        "y_train": screen_y,
        "x_val": normalized_val_x,
        "y_val": normalized_val_y,
        "x_test": normalized_test_x,
        "y_test": normalized_test_y,
        "mean": mean,
        "std": std,
    }
    record = {
        "schema_version": 1,
        "dataset": "UCI HIGGS",
        "source_url": "https://archive.ics.uci.edu/dataset/280/higgs",
        "source_path": str(csv_path.resolve()),
        "source_bytes": csv_path.stat().st_size,
        "source_compressed_sha256": file_sha256(csv_path),
        "total_rows": total_rows,
        "features": FEATURES,
        "ranges": {
            "statistics_training": [0, train_end],
            "screen_training": [0, screen_rows],
            "retained_training": [0, retain_train_rows],
            "validation": [train_end, val_end],
            "heldout_test": [val_end, total_rows],
        },
        "normalization": {
            "statistics_dtype": "float64",
            "statistics": "population moments from all training rows only",
            "chunk_rows": chunk_rows,
            "std_floor": 1e-6,
            "application": "(raw_float32 - mean_float32) / std_float32",
            "train_mean_float64": mean64.tolist(),
            "train_std_float64": std64.tolist(),
        },
        "raw_arrays": {k: array_metadata(v) for k, v in raw_arrays.items()},
        "normalized_arrays": {
            k: array_metadata(v) for k, v in normalized_arrays.items()
        },
        "artifacts": {},
    }
    payloads = {
        "screen.npz": {
            k: normalized_arrays[k]
            for k in ("x_train", "y_train", "x_val", "y_val", "mean", "std")
        },
        "heldout-test.npz": {
            k: normalized_arrays[k] for k in ("x_test", "y_test", "mean", "std")
        },
    }
    if retain_train_rows:
        payloads["retained-train.npz"] = {
            "x_train_raw": train_x[:retain_train_rows],
            "y_train": train_y[:retain_train_rows].astype("<i8"),
            "mean": mean,
            "std": std,
        }
        record["retained_training_raw_arrays"] = {
            "x_train_raw": array_metadata(train_x[:retain_train_rows]),
            "y_train": array_metadata(train_y[:retain_train_rows]),
        }
    # Create only after all source validation. mkdir/exclusive opens also reject
    # a concurrent preparer that chose this same destination.
    outdir.mkdir(parents=True, exist_ok=False)
    for name, arrays in payloads.items():
        path = outdir / name
        with path.open("xb") as destination:
            np.savez_compressed(destination, **arrays)
        record["artifacts"][name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "arrays": list(arrays),
        }
    with (outdir / "manifest.json").open("x") as destination:
        json.dump(record, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--train-end", type=int, default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-end", type=int, default=DEFAULT_VAL_END)
    parser.add_argument("--total-rows", type=int, default=DEFAULT_TOTAL_ROWS)
    parser.add_argument("--screen-rows", type=int, default=100_000)
    parser.add_argument("--retain-train-rows", type=int, default=1_000_000)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = vars(parser.parse_args())
    args["outdir"] = args.pop("output_directory")
    print(json.dumps(prepare(**args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
