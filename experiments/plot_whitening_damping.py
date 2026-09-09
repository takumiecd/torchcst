"""Plot paired damping-sweep scores and initial Gram spectra."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    variants = list(data["median"])
    labels = ["Eigen", "1e-2", "1e-4", "1e-6", "1e-8"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for seed in (17, 29, 43):
        runs = [
            next(t for t in data["trials"] if t["variant"] == v and t["seed"] == seed)
            for v in variants
        ]
        for ax, key, scale in [(axes[0], "accuracy", 100), (axes[1], "step_ms", 1)]:
            ax.plot(
                range(5),
                [r[key] * scale if r[key] is not None else float("nan") for r in runs],
                "o-",
                alpha=0.7,
                label=f"Seed {seed}",
            )
        initial = runs[0]["spectra"][0]["rho_quantiles"]
        axes[2].plot(
            [0, 10, 50, 90, 100], initial, "o-", alpha=0.7, label=f"Seed {seed}"
        )
    for ax, title, ylabel in zip(
        axes[:2],
        ["Held-out accuracy", "Warm training step"],
        ["Percent", "Milliseconds"],
    ):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(5), labels)
        ax.set_xlabel("Eigen baseline / Cholesky whitening damping")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[2].set_title("Initial local Gram eigenvalues")
    axes[2].set_xlabel("Percentile over 256 directions")
    axes[2].set_ylabel("Eigenvalue (log scale)")
    axes[2].set_yscale("log")
    axes[2].axhline(0.01, color="black", ls="--", lw=1, label="Old damping 0.01")
    axes[2].legend(fontsize=7)
    fig.suptitle("Independent whitening damping — 3 paired seeds, 512 steps, A100 MIG")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
