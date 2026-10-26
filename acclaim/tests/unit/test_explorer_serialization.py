"""
Unit tests for the acclaim explorer's display-only record augmentation.
"""

from acclaim_explorer.server.serialization import augment_record


ANSWER = (
    "Paris is the capital of France [1]. "
    "The Eiffel Tower was completed in 1889. "
    "Honestly, it is a lovely city."
)
SENT_2 = (ANSWER.index("The Eiffel"), ANSWER.index(" Honestly"))
SENT_3_START = ANSWER.index("Honestly")


def _claim(text, span=(-1, -1)):
    return {
        "claim": {"text": text, "span": list(span)},
        "citation_doc_ids": [],
        "support": {"label": "UNCLEAR", "confidence": 0.0, "reason": ""},
    }


def _record(claims, filter_decisions=None):
    return {
        "answer": ANSWER,
        "documents": [{"doc_id": "doc1", "text": "Paris is the capital of France."}],
        "claims": claims,
        "filter_decisions": filter_decisions or [],
    }


def test_real_span_passes_through():
    rec = augment_record(_record([_claim("Paris is the capital of France.", (0, 35))]))
    claim = rec["claims"][0]
    assert claim["display_span"] == [0, 35]
    assert claim["display_span_source"] == "aligner"
    assert claim["claim"]["span"] == [0, 35]


def test_spanless_claim_placed_by_overlap():
    rec = augment_record(_record([_claim("The Eiffel Tower was finished in 1889.")]))
    claim = rec["claims"][0]
    assert claim["display_span_source"] == "overlap"
    start, end = claim["display_span"]
    assert start == SENT_2[0]
    assert ANSWER[start:end].startswith("The Eiffel Tower")
    # The raw span is left untouched.
    assert claim["claim"]["span"] == [-1, -1]


def test_no_overlap_leaves_claim_unplaced():
    rec = augment_record(_record([_claim("Quantum chromodynamics.")]))
    claim = rec["claims"][0]
    assert claim["display_span"] is None
    assert claim["display_span_source"] is None


def test_dropped_decisions_become_filtered_claims():
    decisions = [
        {
            "claim": {"text": "Paris is the capital of France.", "span": [-1, -1]},
            "keep": True,
            "filter_name": "check_worthiness",
            "category": "COMMON_KNOWLEDGE",
            "reason": "kept",
        },
        {
            "claim": {"text": "Paris is a lovely city.", "span": [-1, -1]},
            "keep": False,
            "filter_name": "check_worthiness",
            "category": "SUBJECTIVE",
            "reason": "Expresses an opinion.",
        },
    ]
    rec = augment_record(_record([], decisions))
    assert len(rec["filtered_claims"]) == 1
    filtered = rec["filtered_claims"][0]
    assert filtered["text"] == "Paris is a lovely city."
    assert filtered["category"] == "SUBJECTIVE"
    assert filtered["reason"] == "Expresses an opinion."
    assert filtered["display_span"][0] == SENT_3_START


def test_record_without_filter_decisions():
    rec = augment_record({"answer": ANSWER, "documents": [], "claims": []})
    assert rec["filtered_claims"] == []
