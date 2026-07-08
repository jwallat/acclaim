"""
Expand sampled ASQA answers into one row per atomic claim for human annotation
of citation-alignment correctness.

For each claim, JaccardCitationAligner has already overwritten claim.span with
the span of the answer sentence it matched, and citation_doc_ids with the
marker(s) inherited from that sentence. This script surfaces both, plus the
full answer (so the annotator can find the claim's *true* source sentence),
so a human can mark whether the predicted citation_doc_ids actually match the
true source sentence's marker(s).

Usage:
    python build_annotation_sheet.py                     # writes claim_pairs.jsonl + annotation_sheet.csv
    python build_annotation_sheet.py --input my_sampled.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_INPUT = _HERE / "sampled_answers.jsonl"
_DEFAULT_OUTPUT = _HERE / "claim_pairs.jsonl"
_DEFAULT_SHEET = _HERE / "annotation_sheet.csv"


def _load_examples(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _matched_sentence_text(answer: str, span: list[int]) -> str:
    start, end = span
    if start < 0 or end < 0 or end > len(answer):
        return ""
    return answer[start:end]


def build_claim_pairs(examples: list[dict]) -> list[dict]:
    pairs = []
    pair_id = 0
    for ex in examples:
        for claim_idx, claim_result in enumerate(ex["claims"]):
            claim = claim_result["claim"]
            pairs.append({
                "pair_id": pair_id,
                "example_id": ex["example_id"],
                "model": ex["model"],
                "question": ex["question"],
                "answer": ex["answer"],
                "claim_text": claim["text"],
                "matched_sentence_text": _matched_sentence_text(ex["answer"], claim["span"]),
                "predicted_citation_doc_ids": claim_result["citation_doc_ids"],
            })
            pair_id += 1
    return pairs


def write_annotation_sheet(pairs: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "example_id",
                "model",
                "question",
                "answer",
                "claim_text",
                "matched_sentence_text",
                "predicted_citation_doc_ids",
                "correct",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair["pair_id"],
                "example_id": pair["example_id"],
                "model": pair["model"],
                "question": pair["question"],
                "answer": pair["answer"],
                "claim_text": pair["claim_text"],
                "matched_sentence_text": pair["matched_sentence_text"],
                "predicted_citation_doc_ids": ",".join(pair["predicted_citation_doc_ids"]),
                "correct": "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Build citation-alignment annotation sheet.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--sheet", type=Path, default=_DEFAULT_SHEET)
    args = parser.parse_args()

    examples = _load_examples(args.input)
    pairs = build_claim_pairs(examples)

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    write_annotation_sheet(pairs, args.sheet)

    print(f"Wrote {len(pairs)} claim rows from {len(examples)} examples to {args.output}")
    print(f"Wrote annotation sheet to {args.sheet}")


if __name__ == "__main__":
    main()
