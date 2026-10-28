"""
Compute agreement between judge predictions and human labels.

Reads annotation_sheet.csv (human_label column filled in) and one judge
predictions file (or every *.jsonl in a directory), joins on pair_id, and prints accuracy, Cohen's kappa,
macro-F1 (with bootstrap 95% CIs), per-label precision/recall/F1,
calibration of the judge's confidence (ECE, Brier), and a confusion matrix.
Results are also written to results/agreement_results.json (or, with
--predictions-dir, to results/agreement_<run>.json per file).

Usage:
    python compute_agreement.py --predictions predictions/jev-1.13_choice_v3.jsonl
    python compute_agreement.py --predictions my_preds.jsonl --human my_labels.csv --output my_results.json
    python compute_agreement.py --predictions-dir predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).parent
_DEFAULT_HUMAN = _HERE / "annotation_sheet.csv"
_DEFAULT_OUTPUT = _HERE / "results" / "agreement_results.json"

LABELS = ["SUPPORTED", "REFUTED", "UNCLEAR"]


def _load_joined(
    human_path: Path, predictions_path: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (prediction records, human labels) joined on pair_id."""
    with open(human_path, newline="", encoding="utf-8") as f:
        human_by_id = {
            int(row["pair_id"]): row["human_label"].strip().upper()
            for row in csv.DictReader(f)
            if row.get("human_label", "").strip()
        }

    with open(predictions_path, encoding="utf-8") as f:
        records_by_id = {
            rec["pair_id"]: rec for rec in (json.loads(line) for line in f if line.strip())
        }

    records, human_labels = [], []
    for pair_id, human in sorted(human_by_id.items()):
        if pair_id in records_by_id:
            human_labels.append(human)
            records.append(records_by_id[pair_id])

    return records, human_labels


def _load_labeled(
    human_path: Path, predictions_path: Path
) -> tuple[list[str], list[str]]:
    """Return (judge_labels, human_labels) joined on pair_id."""
    records, human_labels = _load_joined(human_path, predictions_path)
    return [r["judge_label"].strip().upper() for r in records], human_labels


def accuracy(judge: list[str], human: list[str]) -> float:
    if not human:
        return 0.0
    return sum(j == h for j, h in zip(judge, human)) / len(human)


def cohen_kappa(judge: list[str], human: list[str]) -> float:
    """Cohen's kappa for multi-class categorical labels."""
    n = len(human)
    if n == 0:
        return 0.0

    all_labels = sorted(set(judge) | set(human))
    observed_agreement = sum(j == h for j, h in zip(judge, human)) / n

    judge_counts = Counter(judge)
    human_counts = Counter(human)
    expected_agreement = sum(
        (judge_counts[label] / n) * (human_counts[label] / n) for label in all_labels
    )

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def per_label_metrics(
    judge: list[str], human: list[str]
) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 per label."""
    metrics: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = sum(j == label and h == label for j, h in zip(judge, human))
        fp = sum(j == label and h != label for j, h in zip(judge, human))
        fn = sum(j != label and h == label for j, h in zip(judge, human))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def macro_f1(judge: list[str], human: list[str]) -> float:
    """Unweighted mean F1 over LABELS (REFUTED is rare, so accuracy alone misleads)."""
    per_label = per_label_metrics(judge, human)
    return sum(m["f1"] for m in per_label.values()) / len(per_label)


def bootstrap_ci(
    metric: Callable[[list[str], list[str]], float],
    judge: list[str],
    human: list[str],
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of *metric* over resampled pairs."""
    rng = random.Random(seed)
    n = len(human)
    values = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(metric([judge[i] for i in idx], [human[i] for i in idx]))
    values.sort()
    return values[int(0.025 * n_resamples)], values[int(0.975 * n_resamples) - 1]


def calibration(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> dict[str, float]:
    """
    Top-label calibration of the judge's confidence in its own label.

    Works for both decision models (probability of the chosen label) and
    chat LLMs (self-reported confidence). ``ece`` is the expected
    calibration error over equal-width bins; ``brier`` is the mean squared
    error between confidence and correctness.
    """
    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "brier": 0.0, "mean_confidence": 0.0}
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        bins[min(int(c * n_bins), n_bins - 1)].append(i)
    ece = sum(
        len(b) / n * abs(sum(confidences[i] for i in b) / len(b) - sum(correct[i] for i in b) / len(b))
        for b in bins
        if b
    )
    brier = sum((c - float(ok)) ** 2 for c, ok in zip(confidences, correct)) / n
    return {"ece": ece, "brier": brier, "mean_confidence": sum(confidences) / n}


def confusion_matrix(judge: list[str], human: list[str]) -> dict[str, dict[str, int]]:
    """Rows = human label, columns = judge label."""
    matrix: dict[str, dict[str, int]] = {
        h: {j: 0 for j in LABELS} for h in LABELS
    }
    for j, h in zip(judge, human):
        if h in matrix and j in matrix[h]:
            matrix[h][j] += 1
    return matrix


def _print_confusion_matrix(matrix: dict[str, dict[str, int]]) -> None:
    col_width = 12
    header = " " * col_width + "".join(f"{'pred ' + l:>{col_width}}" for l in LABELS)
    print(header)
    for h in LABELS:
        row = f"{'true ' + h:>{col_width}}" + "".join(
            f"{matrix[h][j]:>{col_width}}" for j in LABELS
        )
        print(row)


def compute_agreement(
    human_path: Path,
    predictions_path: Path,
    output_path: Path = _DEFAULT_OUTPUT,
) -> dict:
    records, human_labels = _load_joined(human_path, predictions_path)
    judge_labels = [r["judge_label"].strip().upper() for r in records]

    if not human_labels:
        print("No labeled rows found — fill in the human_label column first.")
        return {}

    n = len(human_labels)
    acc = accuracy(judge_labels, human_labels)
    kappa = cohen_kappa(judge_labels, human_labels)
    mf1 = macro_f1(judge_labels, human_labels)
    cis = {
        "accuracy": bootstrap_ci(accuracy, judge_labels, human_labels),
        "cohen_kappa": bootstrap_ci(cohen_kappa, judge_labels, human_labels),
        "macro_f1": bootstrap_ci(macro_f1, judge_labels, human_labels),
    }
    calib = calibration(
        [float(r.get("judge_confidence") or 0.0) for r in records],
        [j == h for j, h in zip(judge_labels, human_labels)],
    )
    per_label = per_label_metrics(judge_labels, human_labels)
    matrix = confusion_matrix(judge_labels, human_labels)

    def _ci(key: str) -> str:
        lo, hi = cis[key]
        return f"[{lo:.3f}, {hi:.3f}]"

    print(f"\n=== Judge Agreement Results ({n} labeled pairs) ===\n")
    print(f"  Accuracy:     {acc:.3f} {_ci('accuracy')}  ({sum(j == h for j, h in zip(judge_labels, human_labels))}/{n} correct)")
    print(f"  Cohen's κ:    {kappa:.3f} {_ci('cohen_kappa')}")
    print(f"  Macro-F1:     {mf1:.3f} {_ci('macro_f1')}")
    print(f"  ECE / Brier:  {calib['ece']:.3f} / {calib['brier']:.3f}  (mean confidence {calib['mean_confidence']:.3f})\n")
    print("  Per-label metrics:")
    for label, m in per_label.items():
        print(f"    {label:<10}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("\n  Confusion matrix (rows=human, cols=judge):")
    _print_confusion_matrix(matrix)
    print()

    results = {
        "n_labeled": n,
        "accuracy": acc,
        "cohen_kappa": kappa,
        "macro_f1": mf1,
        "ci95": cis,
        "calibration": calib,
        "per_label": per_label,
        "confusion_matrix": matrix,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute judge vs human agreement.")
    parser.add_argument("--human", type=Path, default=_DEFAULT_HUMAN)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Run against every *.jsonl file in this directory (e.g. predictions/).")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.predictions is None and args.predictions_dir is None:
        parser.error("pass --predictions <file> or --predictions-dir <dir>")

    if args.predictions_dir is not None:
        files = sorted(args.predictions_dir.glob("*.jsonl"))
        if not files:
            print(f"No *.jsonl files found in {args.predictions_dir}")
            return
        for pred_file in files:
            print(f"\n{'='*60}")
            print(f"Model: {pred_file.stem}")
            print(f"{'='*60}")
            output = _HERE / "results" / f"agreement_{pred_file.stem}.json"
            compute_agreement(args.human, pred_file, output)
    else:
        compute_agreement(args.human, args.predictions, args.output)


if __name__ == "__main__":
    main()
