"""Summarize completed learning runs; target times use recorded checkpoints only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(directory: Path) -> dict:
    groups = {}
    for path in sorted(directory.glob("*_seed*.json")):
        row = json.loads(path.read_text())
        method = path.name.split("_seed")[0]
        groups.setdefault(method, []).append(row)
    report = {}
    for method, rows in groups.items():
        report[method] = {
            "seeds": [r["config"]["seed"] for r in rows],
            "steps": [r["config"]["steps"] for r in rows],
            "accuracy_mean_percent": 100
            * statistics.mean(r["final"]["accuracy"] for r in rows),
            "accuracy_std_percent": 100
            * statistics.pstdev(r["final"]["accuracy"] for r in rows),
            "loss_mean": statistics.mean(r["final"]["loss"] for r in rows),
            "elapsed_mean_seconds": statistics.mean(r["elapsed_seconds"] for r in rows),
            "runs": [
                {
                    "seed": r["config"]["seed"],
                    "accuracy_percent": 100 * r["final"]["accuracy"],
                    "loss": r["final"]["loss"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "converged_steps": sum(s["solver_converged"] for s in r["trace"]),
                    "observed_target_seconds": {
                        str(target): next(
                            (
                                c.get("training_seconds", 0)
                                for c in r["checkpoints"]
                                if c["accuracy"] >= target / 100
                            ),
                            None,
                        )
                        for target in (60, 70, 75, 78)
                    },
                }
                for r in rows
            ],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))


if __name__ == "__main__":
    main()
