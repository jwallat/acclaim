"""Add computed display fields to a dict shaped like EvaluationResult.to_dict()."""

from __future__ import annotations

from acclaim.alignment._scoring import token_weights, tokenize, weighted_claim_overlap
from acclaim.citations.parser import find_citation_markers
from acclaim.text_utils import SentenceSpan, split_sentences_with_spans


def _guess_sentence_span(
    claim_text: str,
    sentences: list[SentenceSpan],
    sent_tokens: list[set[str]],
    weights: dict[str, float],
) -> list[int] | None:
    """Span of the sentence with the highest weighted token overlap with
    *claim_text*, or ``None`` if no sentence shares any token with it."""
    claim_tokens = tokenize(claim_text)
    best_score = 0.0
    best: SentenceSpan | None = None
    for sent, tokens in zip(sentences, sent_tokens):
        score = weighted_claim_overlap(claim_tokens, tokens, weights)
        if score > best_score:
            best_score = score
            best = sent
    return [best.start, best.end] if best is not None else None


def augment_record(base: dict) -> dict:
    """Add sentences/citation_markers/citation_to_doc to a dict already shaped
    like EvaluationResult.to_dict() (question, answer, documents, claims,
    metrics, metadata). Used for both the live-query result and each item
    parsed out of a batch run's JSONL file, so both paths render identically.

    Also adds display-only sentence placements for claims the aligner left
    span-less (uncited atomic claims) and for claims dropped by a claim
    filter, guessed via the same weighted token overlap JaccardCitationAligner
    uses. ``claim.span`` itself is never modified.
    """
    answer = base.get("answer") or ""
    documents = base.get("documents") or []
    claims = base.get("claims") or []
    decisions = base.get("filter_decisions") or []
    dropped = [d for d in decisions if not d.get("keep", True)]

    sentences = [s for s in split_sentences_with_spans(answer) if s.text]
    sent_tokens = [tokenize(s.text) for s in sentences]
    # Down-weight tokens shared by many of the answer's claims (kept and
    # filtered alike), as JaccardCitationAligner does.
    all_claim_texts = [c["claim"]["text"] for c in claims] + [
        d["claim"]["text"] for d in dropped
    ]
    weights = token_weights([tokenize(t) for t in all_claim_texts])

    def _place(claim: dict) -> dict:
        span = claim["claim"].get("span") or [-1, -1]
        if span[0] >= 0:
            return {**claim, "display_span": list(span), "display_span_source": "aligner"}
        guess = _guess_sentence_span(claim["claim"]["text"], sentences, sent_tokens, weights)
        return {
            **claim,
            "display_span": guess,
            "display_span_source": "overlap" if guess is not None else None,
        }

    return {
        **base,
        "claims": [_place(c) for c in claims],
        # Claims dropped by a claim filter (never judged, excluded from all
        # metrics), with a guessed sentence placement for hover linkage.
        "filtered_claims": [
            {
                "text": d["claim"]["text"],
                "category": d.get("category"),
                "reason": d.get("reason"),
                "filter_name": d.get("filter_name"),
                "display_span": _guess_sentence_span(
                    d["claim"]["text"], sentences, sent_tokens, weights
                ),
            }
            for d in dropped
        ],
        # Sentence boundaries computed the same way claim spans are (pysbd),
        # so the frontend highlights claims against the same sentences the
        # aligner actually used instead of re-splitting independently.
        "sentences": [
            {"text": s.text, "start": s.start, "end": s.end}
            for s in split_sentences_with_spans(answer)
        ],
        # Citation marker spans (supports "[2]", "[2][3]", and "[2, 3]"
        # styles), so the frontend links/highlights markers from data
        # instead of re-parsing "[N]" text with its own regex.
        "citation_markers": [
            {"start": m.start, "end": m.end, "numbers": m.numbers}
            for m in find_citation_markers(answer)
        ],
        # Derived positionally from document order — matches the convention
        # evaluate() and run_alce_batch_eval.py already use when no explicit
        # citation_to_doc is supplied.
        "citation_to_doc": {str(i): d["doc_id"] for i, d in enumerate(documents, 1)},
    }
