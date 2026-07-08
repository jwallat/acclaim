"""
Sanity checks for the intrinsic eval metrics.

Implements the three "Verification" cases from PLAN.md:

    1. predicted_claims = [whole answer]
       expectation: faithfulness ~ 1.0, atomicity low (one big claim with
       many conjunctions), coverage_sentence ~ 1.0 (the single claim
       trivially entails every sentence).
    2. predicted_claims = sentences from a *different* answer
       expectation: faithfulness ~ 0.
    3. NLI spot-check on 20 hand-picked (premise, hypothesis) pairs that
       exercise negation and numbers.

These are intended for quick interactive use; they do not replace the
full intrinsic eval on real predictions.

Usage:
    python sanity_check.py --input predicted_claims.jsonl --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from library_evals.atomic_claims.intrinsic_eval import (
    AtomicityChecker,
    NliScorer,
    _autodevice,
    score_example,
)


logger = logging.getLogger(__name__)


# 20 NLI probes covering negation, numerical reasoning, and scope.
# Each item: (premise, hypothesis, expected_high_entailment)
NLI_PROBES: list[tuple[str, str, bool]] = [
    # ---- direct paraphrase ----
    ("Paris is the capital of France.", "The capital of France is Paris.", True),
    # ---- contradiction (negation) ----
    ("Paris is the capital of France.", "Paris is not the capital of France.", False),
    ("The Eiffel Tower is in Paris.", "The Eiffel Tower is not in Paris.", False),
    # ---- entailment via specialization ----
    (
        "The Eiffel Tower was completed in 1889 and stands 330 m.",
        "The Eiffel Tower was completed in 1889.",
        True,
    ),
    (
        "The Eiffel Tower was completed in 1889 and stands 330 m.",
        "The Eiffel Tower stands 330 m.",
        True,
    ),
    # ---- numerical contradiction ----
    ("The Eiffel Tower stands 330 m.", "The Eiffel Tower stands 300 m.", False),
    (
        "The film grossed 250 million dollars.",
        "The film grossed 25 million dollars.",
        False,
    ),
    # ---- numerical entailment ----
    ("The film grossed exactly 250 million dollars.", "The film grossed money.", True),
    # ---- temporal contradiction ----
    ("World War II ended in 1945.", "World War II ended in 1939.", False),
    # ---- temporal entailment ----
    ("World War II ended in 1945.", "World War II ended in the 20th century.", True),
    # ---- subject swap (miscoreference) ----
    ("Alice wrote the book.", "Bob wrote the book.", False),
    # ---- attribute swap ----
    ("The cat is on the mat.", "The dog is on the mat.", False),
    # ---- co-reference / paraphrase ----
    ("The president signed the bill.", "The bill was signed by the president.", True),
    # ---- scope: existential vs universal ----
    ("Some apples are red.", "All apples are red.", False),
    ("All apples are red.", "Some apples are red.", True),
    # ---- negation with quantifier ----
    ("No students passed the exam.", "Some students passed the exam.", False),
    # ---- additive numbers ----
    ("She has three cats and two dogs.", "She has three cats.", True),
    ("She has three cats and two dogs.", "She has five cats.", False),
    # ---- comparative ----
    ("Mount Everest is taller than K2.", "K2 is taller than Mount Everest.", False),
    # ---- unrelated ----
    ("The sky is blue.", "Cats chase mice.", False),
]


def run_check_1(
    records: list[dict], nli: NliScorer, atomicity: AtomicityChecker
) -> dict:
    """Predicted claims = [answer-as-single-claim]."""
    rows = []
    for rec in records:
        synth = dict(rec)
        synth["predicted_claims"] = [rec["answer"]] if rec.get("answer") else []
        scores = score_example(
            record=synth,
            nli=nli,
            atomicity=atomicity,
            threshold=0.5,
            metrics={"faithfulness", "coverage", "atomicity"},
        )
        rows.append(scores)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "faithfulness_mean": sum(r.faithfulness_mean for r in rows) / len(rows),
        "coverage_sentence": sum(r.coverage_sentence for r in rows) / len(rows),
        "atomicity": sum(r.atomicity for r in rows) / len(rows),
    }


def run_check_2(
    records: list[dict], nli: NliScorer, atomicity: AtomicityChecker
) -> dict:
    """Predicted claims = predicted claims from *another* example (shuffled)."""
    if len(records) < 2:
        return {"n": 0}
    rows = []
    for i, rec in enumerate(records):
        donor = records[(i + 1) % len(records)]
        synth = dict(rec)
        synth["predicted_claims"] = donor.get("predicted_claims", [])
        if not synth["predicted_claims"]:
            continue
        scores = score_example(
            record=synth,
            nli=nli,
            atomicity=atomicity,
            threshold=0.5,
            metrics={"faithfulness", "coverage", "atomicity"},
        )
        rows.append(scores)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "faithfulness_mean": sum(r.faithfulness_mean for r in rows) / len(rows),
        "coverage_sentence": sum(r.coverage_sentence for r in rows) / len(rows),
    }


def run_check_3(nli: NliScorer) -> dict:
    """Spot-check NLI behaviour on 20 hand-picked pairs."""
    pairs = [(p, h) for p, h, _ in NLI_PROBES]
    probs = nli.score(pairs)
    correct = 0
    rows = []
    for (p, h, expected), prob in zip(NLI_PROBES, probs):
        is_high = prob >= 0.5
        ok = is_high == expected
        correct += int(ok)
        rows.append(
            {
                "premise": p,
                "hypothesis": h,
                "expected_entail": expected,
                "entail_prob": round(prob, 3),
                "correct": ok,
            }
        )
    return {"accuracy": correct / len(rows), "n": len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "predicted_claims.jsonl",
        help="JSONL produced by run_extractor.py — used for checks 1 and 2.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--nli-model", type=str, default="cross-encoder/nli-deberta-v3-large"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-1-2", action="store_true", help="Skip checks 1 and 2.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = args.device or _autodevice()
    nli = NliScorer(
        model_name=args.nli_model, device=device, batch_size=args.batch_size
    )
    atomicity = AtomicityChecker()

    print("\n=== Check 3: NLI probes ===")
    res3 = run_check_3(nli)
    for r in res3["rows"]:
        flag = "OK " if r["correct"] else "BAD"
        print(
            f"  {flag} expect={'E' if r['expected_entail'] else 'C'} "
            f"p={r['entail_prob']:.2f} | {r['premise'][:50]} -> {r['hypothesis'][:50]}"
        )
    print(f"  accuracy: {res3['accuracy']:.2f} ({res3['n']} probes)")

    if args.skip_1_2 or not args.input.exists():
        if not args.input.exists():
            print(f"\n(skipping checks 1+2: {args.input} not found)")
        return

    records = []
    with args.input.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[: args.limit]

    print("\n=== Check 1: predicted = [answer] ===")
    res1 = run_check_1(records, nli, atomicity)
    print(json.dumps(res1, indent=2))
    print(
        "expected: faithfulness ~1.0, coverage_sentence ~1.0, "
        "atomicity low (single multi-fact claim)"
    )

    print("\n=== Check 2: predicted = claims from another example ===")
    res2 = run_check_2(records, nli, atomicity)
    print(json.dumps(res2, indent=2))
    print("expected: faithfulness collapses toward 0")


if __name__ == "__main__":
    main()
