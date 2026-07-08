"""
Compare inter-model agreement across all predictions_*.jsonl files.

For each pair of models, prints:
  - % of rows with identical judge_label
  - Cohen's kappa

Also prints a per-model label distribution.

Usage:
    python compare_models.py
    python compare_models.py --dir path/to/dir
    python compare_models.py --output inter_model_agreement.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).parent
LABELS = ["SUPPORTED", "REFUTED", "UNCLEAR"]


def _load_predictions(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return {
            rec["pair_id"]: rec["judge_label"].strip().upper()
            for rec in (json.loads(line) for line in f if line.strip())
        }


def _model_name(path: Path) -> str:
    name = path.stem  # e.g. "predictions_claude_haiku_45"
    prefix = "predictions_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _aligned_labels(
    a: dict[int, str], b: dict[int, str]
) -> tuple[list[str], list[str]]:
    shared = sorted(set(a) & set(b))
    return [a[i] for i in shared], [b[i] for i in shared]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def compare_models(
    pred_dir: Path = _HERE,
    output_path: Path = _HERE / "inter_model_agreement.json",
) -> dict:
    files = sorted(pred_dir.glob("predictions*.jsonl"))
    if not files:
        print("No predictions*.jsonl files found.")
        return {}

    models = {_model_name(f): _load_predictions(f) for f in files}

    # --- Label distribution per model ---
    print("\n=== Label distribution per model ===\n")
    col = 12
    header = f"{'':30}" + "".join(f"{l:>{col}}" for l in LABELS) + f"{'total':>{col}}"
    print(header)
    dist_results: dict[str, dict[str, int]] = {}
    for name, preds in models.items():
        counts = Counter(preds.values())
        total = sum(counts.values())
        dist_results[name] = dict(counts)
        row = f"{name[:30]:30}" + "".join(
            f"{counts.get(l, 0):>{col}}" for l in LABELS
        ) + f"{total:>{col}}"
        print(row)

    # --- Pairwise agreement ---
    print("\n=== Pairwise inter-model agreement ===\n")
    names = list(models)
    pair_results: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            la, lb = _aligned_labels(models[na], models[nb])
            n = len(la)
            match = sum(x == y for x, y in zip(la, lb))
            pct = match / n if n else 0.0
            kappa = cohen_kappa(la, lb)
            print(f"  {na[:28]:28} vs {nb[:28]:28}  |  {n} pairs  |  {pct:.1%} match  |  κ={kappa:.3f}")
            pair_results.append(
                {"model_a": na, "model_b": nb, "n": n, "pct_match": pct, "cohen_kappa": kappa}
            )
    print()

    results = {"distributions": dist_results, "pairwise": pair_results}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-model judge agreement.")
    parser.add_argument("--dir", type=Path, default=_HERE)
    parser.add_argument("--output", type=Path, default=_HERE / "inter_model_agreement.json")
    args = parser.parse_args()
    compare_models(args.dir, args.output)


if __name__ == "__main__":
    main()
