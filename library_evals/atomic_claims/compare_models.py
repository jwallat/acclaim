"""
Compare atomic-claim intrinsic-eval summaries across multiple models.

Loads every results_<model>_summary.json produced by run_eval.sh /
run_eval_all_models.sh in this directory and prints/writes one comparison
table (rows = model, columns = summary metrics).

Usage:
    python compare_models.py                       # scan this directory
    python compare_models.py --glob "results_*_summary.json"
    python compare_models.py --csv comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_HERE = Path(__file__).parent

METRICS = [
    ("faithfulness_mean", "Faithfulness (mean ent.)"),
    ("faithful_fraction", "Faithful fraction (>= thr)"),
    ("coverage_sentence", "Coverage (sentence-level)"),
    ("coverage_whole", "Coverage (whole-answer)"),
    ("sentence_claim_coverage", "Coverage (>=1 claim/sent)"),
    ("atomicity", "Atomicity"),
    ("non_redundancy", "Non-redundancy"),
]


def _model_name(path: Path) -> str:
    m = re.match(r"results_(.+)_summary\.json$", path.name)
    return m.group(1) if m else path.stem


def _load_summaries(glob: str) -> dict[str, dict]:
    summaries = {}
    for path in sorted(_HERE.glob(glob)):
        with path.open() as f:
            summaries[_model_name(path)] = json.load(f)
    return summaries


def build_table(summaries: dict[str, dict]) -> list[dict]:
    rows = []
    for model, summary in summaries.items():
        row = {"model": model}
        for key, _label in METRICS:
            stat = summary.get(key) or {}
            row[key] = stat.get("mean") if stat.get("mean") is not None else ""
        cc = summary.get("claim_count") or {}
        row["claim_count_mean"] = cc.get("mean", "")
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("no results_*_summary.json files found")
        return
    header = ["model"] + [key for key, _ in METRICS] + ["claim_count_mean"]
    widths = {h: max(len(h), *(len(f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h])) for r in rows)) for h in header}
    print("  ".join(h.ljust(widths[h]) for h in header))
    for r in rows:
        cells = [
            (f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h])).ljust(widths[h])
            for h in header
        ]
        print("  ".join(cells))


def write_csv(rows: list[dict], path: Path) -> None:
    header = ["model"] + [key for key, _ in METRICS] + ["claim_count_mean"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", type=str, default="results_*_summary.json")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    summaries = _load_summaries(args.glob)
    rows = build_table(summaries)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
