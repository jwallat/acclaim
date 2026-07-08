"""Add computed display fields to a dict shaped like EvaluationResult.to_dict()."""

from __future__ import annotations

from acclaim.citations.parser import find_citation_markers
from acclaim.text_utils import split_sentences_with_spans


def augment_record(base: dict) -> dict:
    """Add sentences/citation_markers/citation_to_doc to a dict already shaped
    like EvaluationResult.to_dict() (question, answer, documents, claims,
    metrics, metadata). Used for both the live-query result and each item
    parsed out of a batch run's JSONL file, so both paths render identically.
    """
    answer = base.get("answer") or ""
    documents = base.get("documents") or []
    return {
        **base,
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
