"""Aggregate disjoint CUDA-event intervals and compare uninstrumented trials."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(directory, baseline):
    runs = []
    for seed in (17, 29, 43):
        r = json.loads((directory / f"s{seed}.json").read_text())
        b = json.loads((baseline / f"cst-s{seed}.json").read_text())
        assert r["status"] == "passed" and len(r["rows"]) == 512
        assert all(row["passed"] for row in r["rows"])
        assert r["initial_parameter_sha256"] == b["initial_parameter_sha256"]
        assert r["batch_permutation_sha256"] == b["batch_permutation_sha256"]
        assert [(e["step"], e["correct"]) for e in r["evaluations"]] == [
            (e["step"], e["correct"]) for e in b["evaluations"]
        ]
        for path, sha in b["source_sha256"].items():
            if path.startswith("src/"):
                assert r["source_sha256"][path] == sha
        rows = r["rows"][2:]
        stages = {}
        for row in rows:
            s = {k: v["cuda_ms"] for k, v in row["stages"].items()}
            exclusive = {
                k: s[k]
                for k in (
                    "zero_grad",
                    "forward_loss",
                    "backward",
                    "complete_observation",
                    "transport",
                    "second_expand",
                    "update_solve",
                    "recompression",
                )
            }
            exclusive["first_expand_other"] = s["first_expand"] - s["transport"]
            exclusive["optimizer_other"] = s["optimizer_total"] - sum(
                s[k]
                for k in (
                    "complete_observation",
                    "first_expand",
                    "second_expand",
                    "update_solve",
                    "recompression",
                )
            )
            for k, v in exclusive.items():
                stages.setdefault(k, []).append(v)
        runs.append(
            {
                "seed": seed,
                "warm_median_ms": r["warm_median_ms"],
                "baseline_warm_median_ms": b["warm_median_ms"],
                "relative_timing_change": r["warm_median_ms"] / b["warm_median_ms"] - 1,
                "stage_mean_ms": {k: statistics.mean(v) for k, v in stages.items()},
                "host_stage_mean_ms": {
                    k: statistics.mean(row["stages"][k]["host_ms"] for row in rows)
                    for k in rows[0]["stages"]
                },
                "source_sha256": r["source_sha256"],
                "raw_sha256": hashlib.sha256(
                    (directory / f"s{seed}.json").read_bytes()
                ).hexdigest(),
                "evaluation_correct_counts_match": True,
                "compression_iterations_median": statistics.median(
                    row["compression"]["iterations"] for row in rows
                ),
                "compression_iterations_max": max(
                    row["compression"]["iterations"] for row in rows
                ),
            }
        )
    means = {
        k: statistics.mean(r["stage_mean_ms"][k] for r in runs)
        for k in runs[0]["stage_mean_ms"]
    }
    return {
        "stage_mean_ms": means,
        "stage_share": {k: v / sum(means.values()) for k, v in means.items()},
        "runs": runs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.directory, args.baseline)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    for r in result["runs"]:
        print(r["seed"], r["warm_median_ms"], r["relative_timing_change"])
