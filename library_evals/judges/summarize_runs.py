"""
Summarise every predictions/*.jsonl into one comparison
table (results_summary.md): accuracy, Cohen's κ and macro-F1 with bootstrap
95% CIs, calibration (ECE / Brier), failures, latency and cost.

For decision-model ``noul`` runs, an extra row reports thresholds tuned by
stratified 5-fold cross-validation on the stored raw P(supported) /
P(refuted) answers. The 0.5-threshold row stays the headline number:
thresholds tuned on all 100 pairs would be optimistic.

Usage:
    python summarize_runs.py
    python summarize_runs.py --dir path/to/predictions --output my_summary.md
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from compute_agreement import (
    _load_joined,
    accuracy,
    bootstrap_ci,
    calibration,
    cohen_kappa,
    macro_f1,
)

_HERE = Path(__file__).parent
_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def _noul_label(p_sup: float, p_ref: float, t_sup: float, t_ref: float) -> str:
    if p_sup >= t_sup:
        return "SUPPORTED"
    if p_ref >= t_ref:
        return "REFUTED"
    return "UNCLEAR"


def _cv_noul(
    nouls: list[tuple[float, float]], human: list[str], k: int = 5, seed: int = 0
) -> tuple[list[str], list[tuple[float, float]]]:
    """Out-of-fold labels from thresholds tuned (macro-F1) on the other folds."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    # Stratify so the 4 REFUTED pairs spread across folds.
    for label in sorted(set(human)):
        idx = [i for i, h in enumerate(human) if h == label]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)

    preds: list[str] = [""] * len(human)
    chosen = []
    for f, test_idx in enumerate(folds):
        train_idx = [i for g, fold in enumerate(folds) if g != f for i in fold]
        best = max(
            ((ts, tr) for ts in _GRID for tr in _GRID),
            key=lambda t: macro_f1(
                [_noul_label(*nouls[i], *t) for i in train_idx], [human[i] for i in train_idx]
            ),
        )
        chosen.append(best)
        for i in test_idx:
            preds[i] = _noul_label(*nouls[i], *best)
    return preds, chosen


def _row(name: str, records: list[dict[str, Any]], judge: list[str], human: list[str]) -> dict:
    ci = {m.__name__: bootstrap_ci(m, judge, human) for m in (accuracy, cohen_kappa, macro_f1)}
    calib = calibration(
        [float(r.get("judge_confidence") or 0.0) for r in records],
        [j == h for j, h in zip(judge, human)],
    )
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    return {
        "run": name,
        "n": len(human),
        "accuracy": accuracy(judge, human),
        "kappa": cohen_kappa(judge, human),
        "macro_f1": macro_f1(judge, human),
        "ci": ci,
        "ece": calib["ece"],
        "brier": calib["brier"],
        "failures": sum(str(r.get("judge_reason", "")).startswith("Judge failed") for r in records),
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "total_cost_usd": sum(costs) if costs else None,
        "pred_dist": {l: judge.count(l) for l in ("SUPPORTED", "REFUTED", "UNCLEAR")},
    }


def summarize(directory: Path, human_path: Path) -> list[dict]:
    rows = []
    for pred_file in sorted(directory.glob("*.jsonl")):
        name = pred_file.stem
        records, human = _load_joined(human_path, pred_file)
        if not human:
            continue
        judge = [r["judge_label"].strip().upper() for r in records]
        rows.append(_row(name, records, judge, human))

        answers = [r.get("decision_answers") or {} for r in records]
        if all("supported" in a and "refuted" in a for a in answers):
            nouls = [(a["supported"]["noul"], a["refuted"]["noul"]) for a in answers]
            cv_judge, chosen = _cv_noul(nouls, human)
            row = _row(f"{name} (5-fold CV thresholds)", records, cv_judge, human)
            row["thresholds"] = chosen
            row["ece"] = row["brier"] = None  # confidences belong to the 0.5-threshold labels
            rows.append(row)
    return rows


def _fmt(x: float | None, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _fmt_ci(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def write_markdown(rows: list[dict], output: Path) -> None:
    lines = [
        "| Run | Acc [95% CI] | κ [95% CI] | Macro-F1 [95% CI] | ECE | Brier | Pred S/R/U | Fail | Latency (s) | Cost (USD, total) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["kappa"]):
        d = r["pred_dist"]
        lines.append(
            f"| {r['run']} | {_fmt_ci(r['accuracy'], r['ci']['accuracy'])} "
            f"| {_fmt_ci(r['kappa'], r['ci']['cohen_kappa'])} "
            f"| {_fmt_ci(r['macro_f1'], r['ci']['macro_f1'])} "
            f"| {_fmt(r['ece'])} | {_fmt(r['brier'])} "
            f"| {d['SUPPORTED']}/{d['REFUTED']}/{d['UNCLEAR']} | {r['failures']} "
            f"| {_fmt(r['mean_latency_s'], 2)} | {_fmt(r['total_cost_usd'], 4)} |"
        )
    cv_rows = [r for r in rows if "thresholds" in r]
    if cv_rows:
        lines += ["", "Thresholds (supported, refuted) chosen per CV fold:", ""]
        lines += [f"- {r['run']}: {r['thresholds']}" for r in cv_rows]
    lines += [
        "",
        "Gold: 100 human-labelled pairs (30 SUPPORTED / 4 REFUTED / 66 UNCLEAR). "
        "ECE/Brier measure the judge's confidence in its own label "
        "(decision models: probability; chat LLMs: self-reported confidence).",
    ]
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise all judge runs into one table.")
    parser.add_argument("--dir", type=Path, default=_HERE / "predictions")
    parser.add_argument("--human", type=Path, default=_HERE / "annotation_sheet.csv")
    parser.add_argument("--output", type=Path, default=_HERE / "results_summary.md")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON dump of all rows")
    args = parser.parse_args()

    rows = summarize(args.dir, args.human)
    write_markdown(rows, args.output)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
