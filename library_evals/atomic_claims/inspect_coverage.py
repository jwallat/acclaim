"""
Per-example coverage breakdown.

For a single example, shows each answer-sentence next to:
  - the top-K most lexically-overlapping claims (Jaccard)
  - the resulting NLI(concat(top-K) -> sentence) probability
  - the single-best-claim NLI score (the old, stricter metric)

Use this to debug why an example's coverage_sentence number looks too low.

Usage:
    python -m library_evals.atomic_claims.inspect_coverage \\
        --predictions library_evals/atomic_claims/predicted_claims.jsonl \\
        --alce-index 25 --topk 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from library_evals.atomic_claims.intrinsic_eval import (
    NliScorer,
    _answer_sentences,
    _autodevice,
    _score_sentence_claim_coverage,
    _select_top_k_claims,
)
from acclaim.text_utils import jaccard_similarity, tokenize_text


def _load(predictions: Path, alce_index: int) -> dict:
    for line in predictions.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("alce_index") == alce_index:
            return rec
    raise SystemExit(f"alce_index={alce_index} not in {predictions}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).parent / "predicted_claims.jsonl",
    )
    parser.add_argument("--alce-index", type=int, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--nli-model", type=str, default="cross-encoder/nli-deberta-v3-large"
    )
    args = parser.parse_args()

    rec = _load(args.predictions, args.alce_index)
    answer = (rec.get("answer") or "").strip()
    claims = [c for c in rec.get("predicted_claims", []) if c and c.strip()]
    sentences = _answer_sentences(answer)

    print(f"=== alce_index={args.alce_index}  n_claims={len(claims)}  n_sentences={len(sentences)} ===\n")
    print("QUESTION:", rec.get("question", "")[:300], "\n")
    print("CLAIMS:")
    for i, c in enumerate(claims):
        print(f"  [{i}] {c}")
    print()

    sent_claim_cov, has_claim, assigned = _score_sentence_claim_coverage(
        sentences, claims
    )

    nli = NliScorer(model_name=args.nli_model, device=args.device or _autodevice())

    # Single-claim NLI(c -> s) for every (c, s) so we can show both numbers.
    single_pairs = [(c, s) for s in sentences for c in claims]
    single_flat = nli.score(single_pairs) if single_pairs else []

    # Top-K concat NLI per sentence.
    selected = [_select_top_k_claims(s, claims, args.topk) for s in sentences]
    topk_pairs = [
        (" ".join(claims[i] for i in sel) if sel else "", s)
        for s, sel in zip(sentences, selected)
    ]
    topk_probs = nli.score(topk_pairs) if topk_pairs else []

    cov_topk = 0
    cov_single = 0
    for j, sentence in enumerate(sentences):
        sel = selected[j]
        topk_p = topk_probs[j]

        chunk = single_flat[j * len(claims) : (j + 1) * len(claims)]
        best_single = max(chunk) if chunk else 0.0
        best_idx = chunk.index(best_single) if chunk else -1

        cov_topk += int(topk_p >= args.threshold)
        cov_single += int(best_single >= args.threshold)

        assigned_claims = [i for i, a in enumerate(assigned) if a == j]
        print(f"--- sentence [{j}] -----------------------------------------------")
        print(f"  S: {sentence}")
        print(
            f"  has-claim (>=1 claim assigned by Jaccard): "
            f"{'YES' if has_claim[j] else 'no'}  "
            f"(assigned claim ids: {assigned_claims})"
        )
        s_toks = tokenize_text(sentence)
        print(
            f"  top-{args.topk} claims (Jaccard): "
            + ", ".join(
                f"[{i}] j={jaccard_similarity(s_toks, tokenize_text(claims[i])):.2f}"
                for i in sel
            )
        )
        for i in sel:
            print(f"      [{i}] {claims[i]}")
        print(f"  NLI(concat top-{args.topk} -> sentence): {topk_p:.3f}  "
              f"{'COVERED' if topk_p >= args.threshold else 'not covered'}")
        if best_idx >= 0:
            print(
                f"  NLI(single best claim [{best_idx}] -> sentence): {best_single:.3f}  "
                f"({'covered' if best_single >= args.threshold else 'not covered'})"
            )
        print()

    if sentences:
        print(
            f"coverage(top-{args.topk}) = {cov_topk}/{len(sentences)} = "
            f"{cov_topk / len(sentences):.3f}"
        )
        print(
            f"coverage(single best)  = {cov_single}/{len(sentences)} = "
            f"{cov_single / len(sentences):.3f}"
        )
        print(
            f"sentence_claim_coverage (>=1 claim/sent) = "
            f"{sum(has_claim)}/{len(sentences)} = {sent_claim_cov:.3f}"
        )


if __name__ == "__main__":
    main()
