"""Condense parameter RMS trials, retaining failed-run censoring."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from experiments.accuracy_targets import summarize_targets


def summarize(directory, baseline, transported=None):
    methods = {}
    modes = ["baseline", "raw", "alpha_diagonal"]
    if transported is not None:
        modes.append("transported_block")
    for mode in modes:
        runs = []
        for seed in (17, 29, 43):
            path = (
                (baseline / f"cst-s{seed}.json")
                if mode == "baseline"
                else directory / f"{mode}-s{seed}.json"
            )
            if mode == "transported_block":
                path = transported / f"{mode}-s{seed}.json"
            r = json.loads(path.read_text())
            assert r["status"] in ("passed", "failed")
            assert r["config"]["steps"] == 512 and r["config"]["eval_every"] == 8
            assert r["protocol"]["lr"] == 0.05
            if mode != "baseline":
                b = json.loads((baseline / f"cst-s{seed}.json").read_text())
                assert r["initial_parameter_sha256"] == b["initial_parameter_sha256"]
                assert r["batch_permutation_sha256"] == b["batch_permutation_sha256"]
                for p, sha in b["source_sha256"].items():
                    if p.startswith("src/"):
                        assert r["source_sha256"][p] == sha
            assert r["target_times"] == summarize_targets(
                r["evaluations"], [0.75, 0.8, 0.85], 3
            )
            warm = [x for x in r["rows"][2:] if x.get("passed")]
            r["valid_updates"] = sum(bool(x.get("passed")) for x in r["rows"])
            r["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            r["best_accuracy"] = max(x["accuracy"] for x in r["evaluations"])
            r["observed_peak_mib"] = max(
                x["cumulative_peak_allocated_mib"] for x in r["evaluations"]
            )
            r["compression_iterations_median"] = (
                statistics.median(x["compression"]["iterations"] for x in warm)
                if warm
                else None
            )
            r.pop("rows")
            runs.append(r)
        completed = [r for r in runs if r["status"] == "passed"]
        aggregate = {"completed": len(completed), "total": 3}
        for key in (
            "final_accuracy",
            "warm_median_ms",
            "warm_update_median_ms",
            "observed_peak_mib",
            "compression_iterations_median",
        ):
            aggregate[key] = (
                statistics.median(r[key] for r in completed) if completed else None
            )
        aggregate["targets"] = {}
        for target in (".75", ".8", ".85"):
            key = str(float(target))
            aggregate["targets"][key] = {}
            for name in ("first_observed", "consecutive_confirmed"):
                events = [
                    r["target_times"][key][name]
                    for r in runs
                    if r["target_times"][key][name]
                ]
                if name == "consecutive_confirmed":
                    events = [e["confirmation"] for e in events]
                aggregate["targets"][key][name] = {
                    "reached": len(events),
                    "median_seconds": statistics.median(
                        e["training_seconds"] for e in events
                    )
                    if events
                    else None,
                }
        methods[mode] = {"aggregate": aggregate, "runs": runs}
    return methods


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("--transported", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.directory, a.baseline, a.transported)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mode, r in result.items():
        print(mode, json.dumps(r["aggregate"]))
        for x in r["runs"]:
            print(
                x["config"]["seed"],
                x["status"],
                x.get("final_accuracy"),
                x.get("error"),
                x["best_accuracy"],
            )
