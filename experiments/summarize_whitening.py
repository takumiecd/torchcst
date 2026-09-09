"""Summarize paired public local-Adam whitening runs, preserving failures."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    trials = []
    for seed in (17, 29, 43):
        pair = [
            json.loads((directory / f"{mode}-s{seed}.json").read_text())
            for mode in ("eigen", "cholesky")
        ]
        for key in (
            "initial_parameter_sha256",
            "batch_permutation_sha256",
            "source_sha256",
        ):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"unmatched seed {seed}: {key}")
        for mode, run in zip(("eigen", "cholesky"), pair):
            trial = {
                "mode": mode,
                "seed": seed,
                "status": run["status"],
                "steps": len(run["rows"]),
                "error": run.get("error"),
                "initial_parameter_sha256": run["initial_parameter_sha256"],
                "batch_permutation_sha256": run["batch_permutation_sha256"],
            }
            if run["status"] == "passed":
                assert len(run["rows"]) == 512 and all(r["passed"] for r in run["rows"])
                trial.update(
                    accuracy=run["final_accuracy"],
                    step_ms=run["warm_median_ms"],
                    update_ms=run["warm_update_median_ms"],
                    peak_mib=run["warm_peak_allocated_mib"],
                )
                for stage in (
                    "whitening",
                    "local_second_expand",
                    "optimizer_total",
                    "local_recompression",
                ):
                    for clock in ("cuda_ms", "host_ms"):
                        trial[f"{stage}_{clock}"] = statistics.median(
                            r["stages"][stage][clock] for r in run["rows"][2:]
                        )
                trial["evaluations"] = run["evaluations"]
            trials.append(trial)
    metrics = (
        "accuracy",
        "step_ms",
        "update_ms",
        "peak_mib",
        "whitening_cuda_ms",
        "whitening_host_ms",
        "local_second_expand_cuda_ms",
        "optimizer_total_cuda_ms",
    )
    summary = {}
    for mode in ("eigen", "cholesky"):
        good = [t for t in trials if t["mode"] == mode and t["status"] == "passed"]
        summary[mode] = {
            "passed": len(good),
            **{
                key: statistics.median(t[key] for t in good) if good else None
                for key in metrics
            },
        }
    return {
        "gpu": run["gpu"],
        "torch": run["torch"],
        "source_sha256": run["source_sha256"],
        "protocol": {
            "seeds": [17, 29, 43],
            "steps": 512,
            "atoms": 64,
            "batch": 128,
            "lr": 0.05,
            "betas": [0.9, 0.99],
            "first_moment_damping": 0.01,
            "update_damping": 0.01,
            "warmup_excluded_steps": 2,
            "stage_timing": True,
        },
        "trials": trials,
        "median": summary,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.directory)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["median"], indent=2))


if __name__ == "__main__":
    main()
