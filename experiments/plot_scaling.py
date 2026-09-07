"""Render the runtime scaling results (requires matplotlib)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in args.input.glob("*.json")]
    failures = [r for r in rows if r.get("status") == "failed"]
    rows = [r for r in rows if "median_ms_per_step" in r]
    indexed = {(r["inputs"], r["outputs"], r["atoms"], r["method"]): r for r in rows}
    widths = sorted({r["inputs"] for r in rows if r["inputs"] == r["outputs"]})
    styles = {
        "dense_adam": ("Dense + Adam", "#597087"),
        "cst_adam": ("CST + Adam", "#27a37c"),
        "cst_ray1": ("CST + Ray 1", "#c35f37"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for method, (label, color) in styles.items():
        key_k = None if method == "dense_adam" else 64
        selected = [
            indexed[(n, n, key_k, method)]
            for n in widths
            if (n, n, key_k, method) in indexed
        ]
        axes[0].plot(
            [r["inputs"] for r in selected],
            [r["median_ms_per_step"] for r in selected],
            "o-",
            label=label,
            color=color,
        )
        axes[2].plot(
            [r["inputs"] for r in selected],
            [r["peak_allocated_mib"] for r in selected],
            "o-",
            label=label,
            color=color,
        )
    for method in ("cst_adam", "cst_ray1"):
        selected = sorted(
            (
                r
                for r in rows
                if r["inputs"] == r["outputs"] == 256 and r["method"] == method
            ),
            key=lambda r: r["atoms"],
        )
        label, color = styles[method]
        axes[1].plot(
            [r["atoms"] for r in selected],
            [r["median_ms_per_step"] for r in selected],
            "o-",
            label=label,
            color=color,
        )
    dense = indexed[(256, 256, None, "dense_adam")]
    axes[1].axhline(
        dense["median_ms_per_step"],
        color=styles["dense_adam"][1],
        linestyle="--",
        label="Dense + Adam (fixed)",
    )
    for ax, title, xlabel, ylabel in zip(
        axes,
        (
            "Width scaling: fixed K=64",
            "Atom scaling: fixed 256 x 256",
            "Peak tensor memory: fixed K=64",
        ),
        ("Input = output width", "Atoms K (4 parameters each)", "Input = output width"),
        (
            "Milliseconds / training step",
            "Milliseconds / training step",
            "MiB allocated",
        ),
    ):
        ax.set(xscale="log", yscale="log", title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in (axes[0], axes[2]):
        ax.set_xticks(widths, labels=[str(n) for n in widths])
    axes[1].set_xticks([16, 32, 64, 128], labels=["16", "32", "64", "128"])
    if failures:
        axes[0].text(
            0.03,
            0.88,
            "Ray 1 at 1024: allocator failure",
            transform=axes[0].transAxes,
            color="#c35f37",
            fontsize=9,
        )
    fig.suptitle(
        "A100 MIG 3g.40gb | batch 128 | warmed float32 training | median of 3 blocks\nRay 1 uses a benchmark-only expanded derivative-cache limit",
        fontsize=12,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "scaling.png", dpi=180)
    fig.savefig(args.output / "scaling.svg")


if __name__ == "__main__":
    main()
