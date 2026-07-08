"""
Sample 100 (claim, single-doc) pairs from the atomic claims experiment output.

Each pair combines one atomic claim with one document from the same example,
giving the judge a single document to evaluate against. Pairs are sampled
with stratification across source examples for topic diversity.

Also writes annotation_sheet.csv for human labeling (fill in human_label column).

Usage:
    python sample_pairs.py                           # writes pairs.jsonl + annotation_sheet.csv
    python sample_pairs.py --output my_pairs.jsonl  # custom output path
    python sample_pairs.py --n 50 --seed 7          # fewer pairs, different seed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_INPUT = _HERE.parent / "atomic_claims" / "predicted_claims.jsonl"
_DEFAULT_OUTPUT = _HERE / "pairs.jsonl"
_DEFAULT_SHEET = _HERE / "annotation_sheet.csv"


def _load_examples(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_pairs(examples: list[dict]) -> dict[int, list[dict]]:
    """Return claim × doc cross-product per example, keyed by alce_index."""
    groups: dict[int, list[dict]] = {}
    for ex in examples:
        claims = ex.get("predicted_claims", [])
        docs = ex.get("docs", [])
        if not claims or not docs:
            continue
        pairs = [
            {
                "alce_index": ex["alce_index"],
                "claim": claim,
                "doc_id": str(doc["id"]),
                "doc_title": doc.get("title", ""),
                "doc_text": doc["text"],
            }
            for claim in claims
            for doc in docs
        ]
        groups[ex["alce_index"]] = pairs
    return groups


def sample_pairs(
    input_path: Path = _DEFAULT_INPUT,
    n: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Return *n* stratified (claim, doc) pairs from *input_path*."""
    rng = random.Random(seed)
    examples = _load_examples(input_path)
    groups = _build_pairs(examples)

    if not groups:
        raise ValueError(f"No valid pairs found in {input_path}")

    # Distribute quota as evenly as possible across examples.
    n_groups = len(groups)
    base_quota = n // n_groups
    remainder = n % n_groups

    selected: list[dict] = []
    keys = sorted(groups)  # deterministic order
    for i, key in enumerate(keys):
        quota = base_quota + (1 if i < remainder else 0)
        pool = groups[key]
        chosen = rng.sample(pool, min(quota, len(pool)))
        selected.extend(chosen)

    # If total < n (because some groups were smaller than their quota),
    # fill the gap with random samples from the full unused pool.
    print(f"Selected {len(selected)} pairs from {n_groups} groups with base quota {base_quota} and remainder {remainder}.")
    print(f"Groups with fewer pairs than their quota: {[key for key in keys if len(groups[key]) < (base_quota + (1 if keys.index(key) < remainder else 0))]}")
    if len(selected) < n:
        selected_set = {(p["alce_index"], p["claim"], p["doc_id"]) for p in selected}
        unused = [
            p
            for pairs in groups.values()
            for p in pairs
            if (p["alce_index"], p["claim"], p["doc_id"]) not in selected_set
        ]
        rng.shuffle(unused)
        selected.extend(unused[: n - len(selected)])

    rng.shuffle(selected)
    return [{"pair_id": i, **p} for i, p in enumerate(selected)]


def write_annotation_sheet(pairs: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "claim", "doc_id", "doc_title", "doc_text_truncated", "human_label"],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair["pair_id"],
                "claim": pair["claim"],
                "doc_id": pair["doc_id"],
                "doc_title": pair["doc_title"],
                "doc_text_truncated": pair["doc_text"][:300],
                "human_label": "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample claim-doc pairs for judge eval.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--sheet", type=Path, default=_DEFAULT_SHEET)
    parser.add_argument("--n", type=int, default=100, help="Number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = sample_pairs(args.input, args.n, args.seed)

    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    write_annotation_sheet(pairs, args.sheet)

    print(f"Wrote {len(pairs)} pairs to {args.output}")
    print(f"Wrote annotation sheet to {args.sheet}")
    alce_indices = {p["alce_index"] for p in pairs}
    print(f"Covering {len(alce_indices)} source examples: {sorted(alce_indices)}")


if __name__ == "__main__":
    main()
