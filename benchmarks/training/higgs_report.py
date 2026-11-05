"""Summarize the one-seed, one-epoch HIGGS middle-layer screening campaign.

Only ``cells/grid-*/learning.json`` are observations. Qualification and learning
rate selection cells are excluded. Comparisons require matching data, budgets,
seed, precision and hardware/runtime; incomplete campaigns remain labelled so.
Matplotlib is an optional reporting dependency and is imported only for plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

_PAIRED_FIELDS = (
    "cohort",
    "width",
    "seed",
    "mode",
    "training_rows",
    "validation_rows",
    "batch_size",
    "dataset_sha256",
    "total_updates",
    "gpu",
    "torch",
    "cuda",
    "cpu_math_threads",
    "TF32",
)
_RUNTIME_FIELDS = ("gpu", "torch", "cuda", "cpu_math_threads", "TF32")
_METRIC_FIELDS = ("loss", "accuracy", "roc_auc")
_LIMITATION = (
    "Single-seed, one-epoch screening on a fixed validation subset; no test "
    "evaluation, uncertainty estimates, noninferiority or minimum-density claim. "
    "Dense and CST use different initializers. Learning rates selected on a "
    "short N1024 screen are reused across widths and densities."
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return value


def _read_cell(path):
    data = json.loads(path.read_text())
    if data["kind"] not in ("cst", "dense"):
        raise ValueError(f"unknown grid arm: {path}")
    if len(data["epochs"]) != 1:
        raise ValueError("this report is restricted to one-epoch screening cells")
    if data.get("all_layers_trainable") is not True:
        raise ValueError(f"grid cell did not train all layers: {path}")
    epoch = data["epochs"][-1]
    if epoch["updates"] != data["total_updates"]:
        raise ValueError(f"inconsistent update budget: {path}")
    if epoch["validation"]["examples"] != data["validation_rows"]:
        raise ValueError(f"inconsistent validation size: {path}")
    for name in _METRIC_FIELDS:
        if not math.isfinite(epoch["validation"][name]):
            raise ValueError(f"non-finite validation {name}: {path}")
    for name in (
        "total_updates",
        "training_seconds",
        "adoption_seconds",
        "training_peak_allocated_bytes",
        "training_peak_reserved_bytes",
    ):
        _positive(data[name], name)
    width = data["width"]
    middle = 4 * data["atom_count"] if data["kind"] == "cst" else width * width
    # Exact inventory for the declared 28-N-N-2 bias-free-middle experiment.
    if data["trainable_parameters"] != middle + 31 * width + 2:
        raise ValueError(f"unexpected HIGGS parameter inventory: {path}")
    if data["kind"] == "cst":
        if data["atom_count"] != width * width // data["divisor"]:
            raise ValueError(f"inconsistent atom count: {path}")
        if data["alpha"] != 1 / data["divisor"]:
            raise ValueError(f"inconsistent atom density: {path}")
    return data


def build_summary(results):
    """Read measured grid cells and produce validated, comparable flat rows."""
    results = Path(results)
    campaign_path = results / "campaign.json"
    campaign = json.loads(campaign_path.read_text())
    paths = sorted((results / "cells").glob("grid-*/learning.json"))
    if not paths:
        raise ValueError(f"no completed grid learning results in {results}")
    cells = [(path, _read_cell(path)) for path in paths]
    recorded_names = {
        cell["name"]
        for cell in campaign.get("completed", [])
        if cell["name"].startswith("grid-")
    }
    actual_names = {path.parent.name for path, _ in cells}
    if campaign["status"] == "complete_screen_only" and recorded_names != actual_names:
        raise ValueError("completed campaign inventory differs from grid result files")
    first_data = cells[0][1]
    for path, data in cells[1:]:
        mismatch = [
            name
            for name in _PAIRED_FIELDS
            if name != "width" and data[name] != first_data[name]
        ]
        if mismatch:
            raise ValueError(
                f"grid conditions differ for {path.parent.name}: {mismatch}"
            )
    seeds = sorted({data["seed"] for _, data in cells})
    if len(seeds) != 1:
        raise ValueError("this single-seed report cannot combine different seeds")
    baselines = {}
    for path, data in cells:
        if data["kind"] == "dense":
            key = (data["width"], data["seed"])
            if key in baselines:
                raise ValueError(f"duplicate dense baseline for {key}: {path}")
            baselines[key] = data
    observations, identities = [], set()
    for path, data in cells:
        key = (data["width"], data["seed"])
        identity = (
            *key,
            data["kind"],
            data["divisor"] if data["kind"] == "cst" else None,
        )
        if identity in identities:
            raise ValueError(f"duplicate grid observation: {identity}")
        identities.add(identity)
        dense = baselines.get(key)
        if dense is not None:
            mismatch = [name for name in _PAIRED_FIELDS if data[name] != dense[name]]
            if mismatch:
                raise ValueError(
                    f"cannot pair {path.parent.name} with dense: {mismatch}"
                )
            if data["memory_scope"] != dense["memory_scope"]:
                raise ValueError(f"memory scopes differ for {path.parent.name}")
            if data["timing_scope"] != dense["timing_scope"]:
                raise ValueError(f"timing scopes differ for {path.parent.name}")
        metrics = data["epochs"][-1]["validation"]
        row = {
            "cell": path.parent.name,
            "kind": data["kind"],
            "width": data["width"],
            "seed": data["seed"],
            "epochs": len(data["epochs"]),
            "alpha_requested": data["alpha"] if data["kind"] == "cst" else None,
            "alpha_actual": data["atom_count"] / data["width"] ** 2
            if data["kind"] == "cst"
            else None,
            "atom_count": data["atom_count"],
            "middle_scalar_parameters": 4 * data["atom_count"]
            if data["kind"] == "cst"
            else data["width"] ** 2,
            "middle_scalar_parameter_ratio": 4 * data["atom_count"] / data["width"] ** 2
            if data["kind"] == "cst"
            else 1.0,
            "total_trainable_parameters": data["trainable_parameters"],
            "lr": data["lr"],
            "training_rows": data["training_rows"],
            "validation_rows": data["validation_rows"],
            "batch_size": data["batch_size"],
            "total_updates": data["total_updates"],
            "graph_replays": data["graph_replays"],
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_roc_auc": metrics["roc_auc"],
            "training_seconds": data["training_seconds"],
            "adoption_seconds": data["adoption_seconds"],
            "mean_full_step_ms": 1000
            * data["training_seconds"]
            / data["total_updates"],
            "mean_adoption_ms": 1000 * data["adoption_seconds"] / data["total_updates"],
            "peak_allocated_bytes": data["training_peak_allocated_bytes"],
            "peak_reserved_bytes": data["training_peak_reserved_bytes"],
            "peak_allocated_mib": data["training_peak_allocated_bytes"] / 2**20,
            "peak_reserved_mib": data["training_peak_reserved_bytes"] / 2**20,
            "paired_dense_available": dense is not None,
            "dataset_sha256": data["dataset_sha256"],
            "initialization_sha256": data["initialization_sha256"],
            "learning_json_sha256": _sha256(path),
            **{field: data[field] for field in _RUNTIME_FIELDS},
        }
        for metric in _METRIC_FIELDS:
            row[f"validation_{metric}_minus_dense"] = (
                metrics[metric] - dense["epochs"][-1]["validation"][metric]
                if dense is not None
                else None
            )
        row["validation_accuracy_minus_dense_pp"] = (
            100 * row["validation_accuracy_minus_dense"] if dense is not None else None
        )
        for output, numerator, denominator in (
            ("total_parameter_ratio", "trainable_parameters", "trainable_parameters"),
            (
                "allocated_ratio",
                "training_peak_allocated_bytes",
                "training_peak_allocated_bytes",
            ),
            (
                "reserved_ratio",
                "training_peak_reserved_bytes",
                "training_peak_reserved_bytes",
            ),
            ("full_step_time_ratio", "training_seconds", "training_seconds"),
            ("adoption_time_ratio", "adoption_seconds", "adoption_seconds"),
        ):
            # Paired update counts are checked above, so total-time ratios equal
            # per-update ratios without assuming a fixed number of steps.
            row[output] = data[numerator] / dense[denominator] if dense else None
        observations.append(row)
    observations.sort(
        key=lambda row: (
            row["width"],
            row["kind"] != "dense",
            row["alpha_requested"] or 0,
        )
    )
    runtimes = {
        json.dumps({field: row[field] for field in _RUNTIME_FIELDS}, sort_keys=True)
        for row in observations
    }
    environment_path = results / "environment.json"
    environment = (
        json.loads(environment_path.read_text()) if environment_path.is_file() else {}
    )
    return {
        "schema_version": 1,
        "campaign_status": campaign["status"],
        "complete_screen": campaign["status"] == "complete_screen_only",
        "results_directory": str(results.resolve()),
        "campaign_json_sha256": _sha256(campaign_path),
        "campaign_recorded_grid_cells": sorted(recorded_names),
        "result_grid_cells": sorted(actual_names),
        "scope": _LIMITATION,
        "plot_label": "HIGGS: single-seed, 1-epoch screening; B=4; no test evaluation",
        "seeds": seeds,
        "widths": sorted({row["width"] for row in observations}),
        "dataset_sha256s": sorted({row["dataset_sha256"] for row in observations}),
        "learning_rates": {
            kind: sorted({row["lr"] for row in observations if row["kind"] == kind})
            for kind in ("dense", "cst")
        },
        "selected_lr": campaign.get("selected_lr", {}),
        "runtimes": [json.loads(value) for value in sorted(runtimes)],
        "host_environment": {
            key: environment[key]
            for key in ("python", "platform")
            if key in environment
        },
        "memory_scopes": sorted({data["memory_scope"] for _, data in cells}),
        "timing_scopes": sorted({data["timing_scope"] for _, data in cells}),
        "observations": observations,
    }


def plot_summary(summary, output):
    """Write standalone scientific PNGs without importing plotting in runtime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, PercentFormatter
    from matplotlib.transforms import Bbox

    output = Path(output)
    rows, widths = summary["observations"], summary["widths"]
    colors = plt.get_cmap("tab10")
    densities = sorted({row["alpha_requested"] for row in rows if row["kind"] == "cst"})

    def density_axis(ax):
        ax.set_xscale("log")
        if densities:
            ax.set_xticks(densities)
            ax.xaxis.set_major_locator(FixedLocator(densities))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax.xaxis.set_minor_formatter(NullFormatter())

    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(
            2, len(widths), figsize=(4 * len(widths), 7), squeeze=False
        )
        for column, width in enumerate(widths):
            group = [row for row in rows if row["width"] == width]
            cst = [row for row in group if row["kind"] == "cst"]
            dense = next((row for row in group if row["kind"] == "dense"), None)
            for line, metric, label in (
                (0, "validation_loss", "Validation cross-entropy (lower is better)"),
                (1, "validation_roc_auc", "Validation ROC-AUC (higher is better)"),
            ):
                ax = axes[line, column]
                if cst:
                    ax.plot(
                        [row["alpha_requested"] for row in cst],
                        [row[metric] for row in cst],
                        "o-",
                        color=colors(column),
                        label="CST middle",
                    )
                if dense is not None:
                    ax.axhline(
                        dense[metric], color="black", linestyle="--", label="Dense"
                    )
                density_axis(ax)
                ax.set_xlabel("Atom density alpha")
                ax.set_ylabel(label)
                ax.grid(True, which="both", alpha=0.2)
                ax.legend(frameon=False)
            axes[0, column].set_title(f"N = {width:,}")
        fig.suptitle(summary["plot_label"])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output / "quality.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, field, label in zip(
            axes.flat,
            (
                "full_step_time_ratio",
                "adoption_time_ratio",
                "allocated_ratio",
                "reserved_ratio",
            ),
            (
                "Complete training-step time / dense",
                "Preparation-inclusive time / dense",
                "Peak allocated memory / dense",
                "Peak reserved memory / dense",
            ),
        ):
            for index, width in enumerate(widths):
                group = [
                    row
                    for row in rows
                    if row["width"] == width
                    and row["kind"] == "cst"
                    and row[field] is not None
                ]
                if group:
                    ax.plot(
                        [row["alpha_requested"] for row in group],
                        [row[field] for row in group],
                        "o-",
                        color=colors(index),
                        label=f"N={width:,}",
                    )
            ax.axhline(1, color="black", linestyle="--", label="Dense")
            density_axis(ax)
            ax.set_xlabel("Atom density alpha")
            ax.set_ylabel(label)
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(frameon=False)
        fig.suptitle(
            summary["plot_label"] + "\nDescriptive single-run ratios; lower is better"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(output / "time-memory.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Cross-width absolute costs expose comparisons that matched-width
        # ratios cannot, including a smaller dense model versus a wider CST.
        # Plot observations only: no inferred frontier or uncertainty estimates.
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        plotted = []
        for ax, cost, label in (
            (axes[0], "peak_allocated_mib", "Peak allocated memory (MiB)"),
            (
                axes[1],
                "training_seconds",
                "Training time (seconds; preparation excluded)",
            ),
        ):
            points = []
            for row in rows:
                color = colors(widths.index(row["width"]))
                dense = row["kind"] == "dense"
                ax.scatter(
                    row[cost],
                    row["validation_roc_auc"],
                    marker="X" if dense else "o",
                    s=65 if dense else 38,
                    color=color,
                    edgecolors="black" if dense else "none",
                    linewidths=0.5,
                    zorder=3,
                )
                configuration = (
                    f"N={row['width']:,}\nDense"
                    if dense
                    else f"N={row['width']:,}\nalpha={100 * row['alpha_requested']:g}%"
                )
                points.append((row[cost], row["validation_roc_auc"], configuration))
            ax.set_xlabel(label)
            ax.set_ylabel("Validation ROC-AUC (higher is better)")
            ax.grid(True, alpha=0.2)
            ax.margins(x=0.16, y=0.22)
            plotted.append((ax, points))
        handles = [
            Line2D(
                [], [], color=colors(i), marker="o", linestyle="none", label=f"N={n:,}"
            )
            for i, n in enumerate(widths)
        ] + [
            Line2D(
                [], [], color="black", marker="o", linestyle="none", label="CST middle"
            ),
            Line2D([], [], color="black", marker="X", linestyle="none", label="Dense"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles), frameon=False
        )
        fig.suptitle(
            summary["plot_label"]
            + "\nFixed training budget; absolute costs across all widths"
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.9))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, points in plotted:
            used = []
            bounds = ax.get_window_extent(renderer)
            markers = [
                Bbox.from_bounds(px - 5, py - 5, 10, 10)
                for px, py in ax.transData.transform([(x, y) for x, y, _ in points])
            ]
            for x, y, label in sorted(points):
                trial = ax.annotate(
                    label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=7
                )
                choices = []
                for dx in (6, -6, 25, -25, 45, -45):
                    for dy in (6, -6, 22, -22, 40, -40, 60, -60):
                        trial.set_position((dx, dy))
                        trial.set_ha("left" if dx > 0 else "right")
                        trial.set_va("bottom" if dy > 0 else "top")
                        box = trial.get_window_extent(renderer).padded(2)
                        outside = sum(
                            (
                                max(bounds.x0 - box.x0, 0),
                                max(box.x1 - bounds.x1, 0),
                                max(bounds.y0 - box.y0, 0),
                                max(box.y1 - bounds.y1, 0),
                            )
                        )
                        score = (
                            10_000 * sum(box.overlaps(other) for other in used)
                            + 100 * sum(box.overlaps(marker) for marker in markers)
                            + 1000 * outside
                            + abs(dx)
                            + abs(dy)
                        )
                        choices.append((score, dx, dy, box))
                _, dx, dy, box = min(choices, key=lambda choice: choice[0])
                trial.remove()
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top",
                    fontsize=7,
                    arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.5},
                )
                used.append(box)
        fig.savefig(output / "quality-efficiency.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args):
    summary = build_summary(args.results)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (args.output / "summary.csv").open("x", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(summary["observations"][0])
        )
        writer.writeheader()
        writer.writerows(summary["observations"])
    if not args.no_plots:
        plot_summary(summary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "grid_cells": len(summary["observations"]),
                "campaign_status": summary["campaign_status"],
            }
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
