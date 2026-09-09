"""Summarize the 128-step learning-rate sweep without hiding failed trials."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    trials = []
    sources = None
    for seed in (17, 29, 43):
        reference = json.loads((directory / f"lr0.05-s{seed}.json").read_text())
        for lr in ("0.03", "0.05", "0.07", "0.10"):
            r = json.loads((directory / f"lr{lr}-s{seed}.json").read_text())
            if sources is None:
                sources = r["source_sha256"]
            assert sources == r["source_sha256"]
            for key in ("initial_parameter_sha256", "batch_permutation_sha256"):
                assert r[key] == reference[key]
            t = {
                "lr": float(lr),
                "seed": seed,
                "status": r["status"],
                "error": r.get("error"),
                "valid_steps": sum(row.get("passed", False) for row in r["rows"]),
                "evaluations": r["evaluations"],
                "accuracy": r.get("final_accuracy"),
                "step_ms": r.get("warm_median_ms"),
                "peak_mib": r.get("warm_peak_allocated_mib"),
                "initial_parameter_sha256": r["initial_parameter_sha256"],
                "batch_permutation_sha256": r["batch_permutation_sha256"],
            }
            trials.append(t)
    summary = {}
    for lr in (0.03, 0.05, 0.07, 0.10):
        good = [t for t in trials if t["lr"] == lr and t["status"] == "passed"]
        summary[str(lr)] = {
            "passed": len(good),
            "median_accuracy": statistics.median(t["accuracy"] for t in good)
            if good
            else None,
            "mean_accuracy": statistics.mean(t["accuracy"] for t in good)
            if good
            else None,
            "step_ms": statistics.median(t["step_ms"] for t in good) if good else None,
        }
    return {
        "gpu": r["gpu"],
        "torch": r["torch"],
        "source_sha256": sources,
        "protocol": {
            "steps": 128,
            "atoms": 64,
            "parameters": 256,
            "seeds": [17, 29, 43],
            "whitening_damping": 1e-4,
            "first_moment_damping": 0.01,
            "update_damping": 0.01,
            "kernel": "AmplitudeBandwidthSeparable",
            "input_sigma": 0.25,
            "output_sigma": 0.1,
            "sigma_explore": "infinity",
            "tau": 0.005,
            "temperature": 0.25,
        },
        "summary": summary,
        "trials": trials,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.directory)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
