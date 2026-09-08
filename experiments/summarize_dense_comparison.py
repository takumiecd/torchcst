"""Validate and condense the paired dense/CST MNIST comparison."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from experiments.accuracy_targets import summarize_targets


def summarize(directory):
    trials = {}
    for method in ("dense-adam", "cst-adam", "cst-adam-fast", "cst"):
        trials[method] = []
        for seed in (17, 29, 43):
            path = directory / f"{method}-s{seed}.json"
            report = json.loads(path.read_text())
            assert report["status"] == "passed", path
            assert len(report["rows"]) == 512, path
            assert all(row["passed"] for row in report["rows"]), path
            assert report["config"]["seed"] == seed
            assert report["config"]["eval_every"] == 8
            assert report["target_times"] == summarize_targets(
                report["evaluations"], [0.75, 0.80, 0.85], 3
            )
            report["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["peak_allocated_mib"] = max(
                e["cumulative_peak_allocated_mib"] for e in report["evaluations"]
            )
            report["total_training_seconds"] = sum(r["seconds"] for r in report["rows"])
            del report["rows"]
            trials[method].append(report)
    reference = trials["cst"][0]
    for runs in trials.values():
        for run in runs:
            assert run["source_sha256"] == reference["source_sha256"]
            assert run["torch"] == reference["torch"] and run["gpu"] == reference["gpu"]
            assert {k: v for k, v in run["protocol"].items() if k != "lr"} == {
                k: v for k, v in reference["protocol"].items() if k != "lr"
            }
    for i in range(3):
        assert (
            len({runs[i]["batch_permutation_sha256"] for runs in trials.values()}) == 1
        )
        assert (
            len(
                {
                    trials[m][i]["initial_parameter_sha256"]
                    for m in ("cst", "cst-adam", "cst-adam-fast")
                }
            )
            == 1
        )
    aggregate = {}
    for method, runs in trials.items():
        aggregate[method] = {
            key: statistics.median(r[key] for r in runs)
            for key in (
                "warm_median_ms",
                "peak_allocated_mib",
                "total_training_seconds",
                "final_accuracy",
            )
        }
        aggregate[method]["targets"] = {}
        for target in reference["target_times"]:
            aggregate[method]["targets"][target] = {}
            for mode in ("first_observed", "consecutive_confirmed"):
                reached = [
                    r["target_times"][target][mode]
                    for r in runs
                    if r["target_times"][target][mode] is not None
                ]
                if mode == "consecutive_confirmed":
                    reached = [r["confirmation"] for r in reached]
                aggregate[method]["targets"][target][mode] = {
                    "reached": len(reached),
                    "total": 3,
                    "median_training_seconds": statistics.median(
                        r["training_seconds"] for r in reached
                    )
                    if reached
                    else None,
                }
    return {"aggregate": aggregate, "trials": trials}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
