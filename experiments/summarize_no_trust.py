"""Summarize the unconstrained update sweep against the prior radius baseline."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from experiments.accuracy_targets import summarize_targets


def summarize(directory, baseline):
    methods = {}
    for name in ("trust", "0.005", "0.01", "0.05"):
        runs = []
        for seed in (17, 29, 43):
            path = (
                baseline / f"local_both-s{seed}.json"
                if name == "trust"
                else directory / f"lr{name}-s{seed}.json"
            )
            r = json.loads(path.read_text())
            assert r["status"] in ("passed", "failed")
            b = json.loads((baseline / f"local_both-s{seed}.json").read_text())
            assert r["initial_parameter_sha256"] == b["initial_parameter_sha256"]
            assert r["batch_permutation_sha256"] == b["batch_permutation_sha256"]
            assert r["target_times"] == summarize_targets(
                r["evaluations"], [0.75, 0.8, 0.85], 3
            )
            r["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            valid = [x for x in r["rows"] if x.get("passed")]
            r["valid_steps"] = len(valid)
            r["max_displacement_norm"] = max(
                (x["displacement_norm"] for x in valid if "displacement_norm" in x),
                default=None,
            )
            r["above_old_radius"] = (
                sum(x.get("displacement_norm", 0) > 0.25 for x in valid)
                if name != "trust"
                else None
            )
            r["peak_mib"] = max(
                x["cumulative_peak_allocated_mib"] for x in r["evaluations"]
            )
            del r["rows"]
            runs.append(r)
        passed = [r for r in runs if r["status"] == "passed"]
        a = {"completed": len(passed), "total": 3}
        for key in (
            "final_accuracy",
            "warm_median_ms",
            "warm_update_median_ms",
            "peak_mib",
        ):
            a[key] = statistics.median(r[key] for r in passed) if passed else None
        a["targets"] = {}
        for t in ("0.75", "0.8", "0.85"):
            a["targets"][t] = {}
            for mode in ("first_observed", "consecutive_confirmed"):
                es = [
                    r["target_times"][t][mode]
                    for r in runs
                    if r["target_times"][t][mode]
                ]
                if mode == "consecutive_confirmed":
                    es = [x["confirmation"] for x in es]
                a["targets"][t][mode] = {
                    "reached": len(es),
                    "median_seconds": statistics.median(
                        x["training_seconds"] for x in es
                    )
                    if es
                    else None,
                }
        methods[name] = {"aggregate": a, "runs": runs}
    return methods


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = summarize(a.directory, a.baseline)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    for name, x in r.items():
        print(name, x["aggregate"])
        for v in x["runs"]:
            print(
                v["config"]["seed"],
                v["status"],
                v.get("final_accuracy"),
                v["max_displacement_norm"],
                v.get("error"),
            )
