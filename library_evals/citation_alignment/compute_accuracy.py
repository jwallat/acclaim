"""
Compute citation-alignment accuracy from the filled-in annotation sheet.

Reads annotation_sheet.csv (correct column filled in with y/n, 1/0, or
similar) and reports the percentage of *labeled* atomic claims whose
predicted citation_doc_ids matched their true source sentence's marker(s),
plus a per-model breakdown for sanity-checking. Rows with a blank correct
column are skipped, so this can be run on partially-annotated sheets to
check progress. Results are also written to accuracy_results.json.

Usage:
    python compute_accuracy.py
    python compute_accuracy.py --sheet my_annotation_sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_SHEET = _HERE / "annotation_sheet.csv"
_DEFAULT_OUTPUT = _HERE / "accuracy_results.json"

_TRUE_VALUES = {"y", "yes", "1", "true", "correct"}
_FALSE_VALUES = {"n", "no", "0", "false", "incorrect"}


def _parse_correct(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognized 'correct' value: {value!r}")


def compute_accuracy(sheet_path: Path = _DEFAULT_SHEET, output_path: Path = _DEFAULT_OUTPUT) -> dict:
    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    labeled_rows = [row for row in rows if (row.get("correct") or "").strip()]
    labels = [_parse_correct(row["correct"]) for row in labeled_rows]
    n = len(labels)
    n_correct = sum(labels)
    accuracy = n_correct / n if n else 0.0

    by_model: dict[str, list[bool]] = {}
    for row, label in zip(labeled_rows, labels):
        by_model.setdefault(row["model"], []).append(label)
    per_model = {
        model: {"n": len(vals), "accuracy": sum(vals) / len(vals) if vals else 0.0}
        for model, vals in sorted(by_model.items())
    }

    print(f"\n=== Citation Alignment Accuracy ({n}/{total} claims labeled so far) ===\n")
    print(f"  Accuracy: {accuracy:.3f}  ({n_correct}/{n} correct)\n")
    print("  Per-model breakdown:")
    for model, stats in per_model.items():
        print(f"    {model:<15}  n={stats['n']:<4}  accuracy={stats['accuracy']:.3f}")
    print()

    results = {
        "n_total": total,
        "n_labeled": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "per_model": per_model,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute citation-alignment accuracy.")
    parser.add_argument("--sheet", type=Path, default=_DEFAULT_SHEET)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    compute_accuracy(args.sheet, args.output)


if __name__ == "__main__":
    main()
