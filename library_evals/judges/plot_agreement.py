"""
Plot accuracy and Cohen's kappa for old vs new prompt, per model.

Usage:
    python plot_agreement.py
    python plot_agreement.py --old old_prompt/ --new . --output comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).parent
_DEFAULT_OLD = _HERE / "old_prompt"
_DEFAULT_NEW = _HERE
_DEFAULT_OUTPUT = _HERE / "prompt_comparison.png"


def _load_results(directory: Path) -> dict[str, dict]:
    results = {}
    for f in directory.glob("agreement_predictions_*.json"):
        model = f.stem.removeprefix("agreement_predictions_")
        with open(f) as fh:
            results[model] = json.load(fh)
    return results


def plot(old_dir: Path, new_dir: Path, output: Path) -> None:
    old = _load_results(old_dir)
    new = _load_results(new_dir)

    models = sorted(set(old) & set(new))
    if not models:
        print("No models found in both directories.")
        return

    old_acc = [old[m]["accuracy"] for m in models]
    new_acc = [new[m]["accuracy"] for m in models]
    old_kappa = [old[m]["cohen_kappa"] for m in models]
    new_kappa = [new[m]["cohen_kappa"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Old prompt vs New prompt — human agreement", fontsize=13, fontweight="bold")

    short_names = [m.replace("-", "-\n") for m in models]

    for ax, old_vals, new_vals, title, ylabel in [
        (ax1, old_acc, new_acc, "Accuracy", "Accuracy"),
        (ax2, old_kappa, new_kappa, "Cohen's κ", "Cohen's κ"),
    ]:
        bars_old = ax.bar(x - width / 2, old_vals, width, label="Old prompt", color="#7eb0d5", edgecolor="white")
        bars_new = ax.bar(x + width / 2, new_vals, width, label="New prompt", color="#fd7f6f", edgecolor="white")

        for bar in bars_old:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#555")
        for bar in bars_new:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#555")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.61, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="κ=0.61 threshold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=_DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=_DEFAULT_NEW)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.old, args.new, args.output)


if __name__ == "__main__":
    main()
