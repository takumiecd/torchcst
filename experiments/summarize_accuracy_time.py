"""Aggregate paired threshold trials and plot all seeds without interpolation."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.accuracy_targets import summarize_targets

METHODS = {
    ("spectral", "pair"): "spectral-pair",
    ("spectral", "jvp_vjp"): "spectral-stream",
    ("krylov", "jvp_vjp"): "krylov-stream",
}
LABELS = {
    "spectral-pair": "Spectral + pair",
    "spectral-stream": "Spectral + JVP/VJP",
    "krylov-stream": "Krylov + JVP/VJP",
}
COLORS = dict(zip(LABELS, ["#0072B2", "#009E73", "#D55E00"]))


def value_range(values):
    available = [v for v in values if v is not None]
    return [min(available), max(available)] if available else None


def aggregate(paths):
    trials = []
    versions = {}
    paired = {}
    seen = set()
    protocol = None
    for path in paths:
        raw = json.loads(path.read_text())
        if raw["status"] == "running":
            raise ValueError(f"incomplete trial: {path}")
        config = raw["config"]
        current_protocol = (
            raw["protocol"],
            {
                k: config[k]
                for k in ("steps", "eval_every", "targets", "consecutive", "atoms")
            },
        )
        if protocol is not None and current_protocol != protocol:
            raise ValueError("trial protocols do not match")
        protocol = current_protocol
        method = METHODS[config["solver"], config["recompression_action"]]
        key = method, config["seed"]
        if key in seen:
            raise ValueError(f"duplicate trial {key}")
        seen.add(key)
        hashes = raw["initial_parameter_sha256"], raw["batch_permutation_sha256"]
        seed = config["seed"]
        if seed in paired and hashes != paired[seed]:
            raise ValueError(f"unpaired initial values or data order: seed {seed}")
        paired[seed] = hashes
        assert raw["target_times"] == summarize_targets(
            raw["evaluations"], config["targets"], config["consecutive"]
        )
        fingerprint = raw.pop("source_sha256")
        version = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()
        versions[version] = fingerprint
        rows = raw.pop("rows")
        raw["method"] = method
        raw["source_version"] = version
        raw["raw_filename"] = path.name
        raw["row_summary"] = {
            "attempted_steps": len(rows),
            "passed_steps": sum(r.get("passed", False) for r in rows),
            "total_training_seconds": sum(r["seconds"] for r in rows),
            "startup_two_updates_seconds": sum(r["seconds"] for r in rows[:2]),
            "peak_training_allocated_mib": max(
                r["memory"]["peak_allocated_mib"] for r in rows
            ),
            "last_attempt": rows[-1],
        }
        trials.append(raw)
    groups = defaultdict(list)
    for trial in trials:
        groups[trial["method"]].append(trial)
    summaries = {}
    for method, runs in groups.items():
        by_target = {}
        for target in runs[0]["target_times"]:
            target_summary = {}
            for kind in ["first_observed", "consecutive_confirmed"]:
                observations = []
                for run in runs:
                    observation = run["target_times"][target][kind]
                    if observation is not None:
                        if kind == "consecutive_confirmed":
                            observation = observation["confirmation"]
                        observations.append(observation)
                target_summary[kind] = {
                    "reached": len(observations),
                    "trials": len(runs),
                    # Conditional medians are descriptive, not a ranking when
                    # another run is censored. Retain all per-seed observations.
                    "reached_only_median_training_seconds": statistics.median(
                        o["training_seconds"] for o in observations
                    )
                    if observations
                    else None,
                    "reached_only_median_wall_seconds": statistics.median(
                        o["wall_seconds"] for o in observations
                    )
                    if observations
                    else None,
                    "reached_only_median_seconds_after_two_updates": statistics.median(
                        o["training_seconds_after_two_updates"] for o in observations
                    )
                    if observations
                    else None,
                    "reached_only_peak_allocated_mib_range": value_range(
                        o["cumulative_peak_allocated_mib"] for o in observations
                    ),
                    "reached_only_warm_peak_allocated_mib_range": value_range(
                        o["cumulative_warm_peak_allocated_mib"] for o in observations
                    ),
                }
            by_target[target] = target_summary
        summaries[method] = {
            "trials": len(runs),
            "failed_trials": sum(r["status"] != "passed" for r in runs),
            "targets": by_target,
        }
    return {
        "source_versions": versions,
        "paired_hashes_verified": True,
        "trials": trials,
        "summary": summaries,
    }


def plot(result, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    styles = {17: "-", 29: "--", 43: ":"}
    for trial in result["trials"]:
        evaluations = trial["evaluations"]
        method = trial["method"]
        for axis, clock in zip(
            axes, ["training_seconds", "training_seconds_after_two_updates"]
        ):
            axis.plot(
                [r[clock] for r in evaluations],
                [100 * r["accuracy"] for r in evaluations],
                color=COLORS[method],
                linestyle=styles[trial["config"]["seed"]],
                linewidth=1.1,
                alpha=0.8,
            )
    for axis in axes:
        for value in [75, 80, 85]:
            axis.axhline(value, color="#888888", linewidth=0.6, alpha=0.6)
        axis.set_ylim(65, 90)
        axis.set_xlim(left=0)
        axis.grid(alpha=0.12)
        axis.set_xlabel("Cumulative training time (s)")
    axes[0].set_ylabel("MNIST held-out accuracy (%)")
    axes[0].set_title("Including first-use capture / compilation")
    axes[1].set_title("Excluding the first two startup updates")
    handles = [Line2D([0], [0], color=COLORS[m], label=LABELS[m]) for m in LABELS]
    handles += [
        Line2D([0], [0], color="#555555", linestyle=styles[s], label=f"Seed {s}")
        for s in styles
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.suptitle(
        "Time to accuracy: 64 atoms, up to 512 updates, evaluation every 8 updates",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for suffix in ["png", "svg"]:
        fig.savefig(destination.with_suffix("." + suffix), dpi=180, bbox_inches="tight")
        if suffix == "svg":
            svg = destination.with_suffix(".svg")
            svg.write_text(
                "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*-s*.json"))
    if len(paths) != 9:
        parser.error("expected three methods times three seeds")
    result = aggregate(paths)
    expected = {(method, seed) for method in LABELS for seed in (17, 29, 43)}
    observed = {
        (trial["method"], trial["config"]["seed"]) for trial in result["trials"]
    }
    if observed != expected or len(result["source_versions"]) != 1:
        parser.error("expected paired methods/seeds and identical experiment sources")
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root / "experiments/accuracy_targets.py"]
    result["aggregation_source_sha256"] = {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }
    if args.archive is not None:
        result["archive"] = {
            "path": str(args.archive),
            "sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "bytes": args.archive.stat().st_size,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot(result, args.output.with_suffix(""))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
