"""
Compute agreement between judge predictions and human labels.

Reads annotation_sheet.csv (human_label column filled in) and predictions.jsonl
(judge output), joins on pair_id, and prints accuracy, Cohen's kappa,
per-label precision/recall/F1, and a confusion matrix.
Results are also written to agreement_results.json.

Usage:
    python compute_agreement.py
    python compute_agreement.py --human my_labels.csv --predictions my_preds.jsonl
    python compute_agreement.py --output my_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_HUMAN = _HERE / "annotation_sheet.csv"
_DEFAULT_PREDICTIONS = _HERE / "predictions.jsonl"
_DEFAULT_OUTPUT = _HERE / "agreement_results.json"

LABELS = ["SUPPORTED", "REFUTED", "UNCLEAR"]


def _load_labeled(
    human_path: Path, predictions_path: Path
) -> tuple[list[str], list[str]]:
    """Return (judge_labels, human_labels) joined on pair_id."""
    with open(human_path, newline="", encoding="utf-8") as f:
        human_by_id = {
            int(row["pair_id"]): row["human_label"].strip().upper()
            for row in csv.DictReader(f)
            if row.get("human_label", "").strip()
        }

    with open(predictions_path, encoding="utf-8") as f:
        judge_by_id = {
            rec["pair_id"]: rec["judge_label"].strip().upper()
            for rec in (json.loads(line) for line in f if line.strip())
        }

    judge_labels, human_labels = [], []
    for pair_id, human in sorted(human_by_id.items()):
        if pair_id in judge_by_id:
            human_labels.append(human)
            judge_labels.append(judge_by_id[pair_id])

    return judge_labels, human_labels


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
    human_path: Path = _DEFAULT_HUMAN,
    predictions_path: Path = _DEFAULT_PREDICTIONS,
    output_path: Path = _DEFAULT_OUTPUT,
) -> dict:
    judge_labels, human_labels = _load_labeled(human_path, predictions_path)

    if not human_labels:
        print("No labeled rows found — fill in the human_label column first.")
        return {}

    n = len(human_labels)
    acc = accuracy(judge_labels, human_labels)
    kappa = cohen_kappa(judge_labels, human_labels)
    per_label = per_label_metrics(judge_labels, human_labels)
    matrix = confusion_matrix(judge_labels, human_labels)

    print(f"\n=== Judge Agreement Results ({n} labeled pairs) ===\n")
    print(f"  Accuracy:     {acc:.3f}  ({sum(j == h for j, h in zip(judge_labels, human_labels))}/{n} correct)")
    print(f"  Cohen's κ:    {kappa:.3f}\n")
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
        "per_label": per_label,
        "confusion_matrix": matrix,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute judge vs human agreement.")
    parser.add_argument("--human", type=Path, default=_DEFAULT_HUMAN)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Run against all predictions*.jsonl files in this directory.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.predictions_dir is not None:
        files = sorted(args.predictions_dir.glob("predictions*.jsonl"))
        if not files:
            print(f"No predictions*.jsonl files found in {args.predictions_dir}")
            return
        for pred_file in files:
            print(f"\n{'='*60}")
            print(f"Model: {pred_file.stem}")
            print(f"{'='*60}")
            output = pred_file.parent / f"agreement_{pred_file.stem}.json"
            compute_agreement(args.human, pred_file, output)
    else:
        predictions = args.predictions or _DEFAULT_PREDICTIONS
        compute_agreement(args.human, predictions, args.output)


if __name__ == "__main__":
    main()
