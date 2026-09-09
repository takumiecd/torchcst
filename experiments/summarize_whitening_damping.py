"""Summarize the paired damping sweep, retaining failed trials and spectra."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory):
    trials = []
    reference_sources = None
    for seed in (17, 29, 43):
        baseline = json.loads((directory / f"eigen-s{seed}.json").read_text())
        for name in (
            "eigen",
            "cholesky-0.01",
            "cholesky-0.0001",
            "cholesky-0.000001",
            "cholesky-0.00000001",
        ):
            run = json.loads((directory / f"{name}-s{seed}.json").read_text())
            if reference_sources is None:
                reference_sources = run["source_sha256"]
            assert run["source_sha256"] == reference_sources
            for key in ("initial_parameter_sha256", "batch_permutation_sha256"):
                assert run[key] == baseline[key]
            trial = {
                "variant": name,
                "seed": seed,
                "status": run["status"],
                "error": run.get("error"),
                "valid_steps": sum(r.get("passed", False) for r in run["rows"]),
                "initial_parameter_sha256": run["initial_parameter_sha256"],
                "batch_permutation_sha256": run["batch_permutation_sha256"],
                "spectra": run["spectra"],
                "evaluations": run["evaluations"],
                "accuracy": None,
                "step_ms": None,
                "whitening_ms": None,
                "peak_mib": None,
            }
            if run["status"] == "passed":
                assert trial["valid_steps"] == 512
                trial.update(
                    accuracy=run["final_accuracy"],
                    step_ms=run["warm_median_ms"],
                    peak_mib=run["warm_peak_allocated_mib"],
                    whitening_ms=statistics.median(
                        r["stages"]["whitening"]["cuda_ms"] for r in run["rows"][2:]
                    ),
                )
            trials.append(trial)
    medians = {}
    for name in dict.fromkeys(t["variant"] for t in trials):
        good = [t for t in trials if t["variant"] == name and t["status"] == "passed"]
        medians[name] = {
            "passed": len(good),
            **{
                key: statistics.median(t[key] for t in good) if good else None
                for key in ("accuracy", "step_ms", "whitening_ms", "peak_mib")
            },
        }
    return {
        "gpu": baseline["gpu"],
        "torch": baseline["torch"],
        "source_sha256": reference_sources,
        "protocol": {
            "steps": 512,
            "seeds": [17, 29, 43],
            "batch": 128,
            "atoms": 64,
            "lr": 0.05,
            "first_moment_damping": 0.01,
            "update_damping": 0.01,
            "spectrum_steps": "0 then every 16, at current model point, outside training timers",
            "quantile_probabilities": [0, 0.1, 0.5, 0.9, 1],
        },
        "median": medians,
        "trials": trials,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = summarize(a.directory)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
    print(json.dumps(r["median"], indent=2))


if __name__ == "__main__":
    main()
