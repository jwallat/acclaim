"""
Unit tests for the LLMCitationAligner (all LLM calls mocked).
"""

import json
from unittest.mock import patch

from acclaim.alignment.llm import LLMCitationAligner
from acclaim.data_models import Claim


TEXT = "Paris is the capital of France [1]. The Eiffel Tower stands 330 m [2]."
CITATION_TO_DOC = {1: "doc-france", 2: "doc-eiffel"}


def _make_aligner(**kwargs) -> LLMCitationAligner:
    return LLMCitationAligner(model="gpt-4o-mini", **kwargs)


def _response(*pairs: tuple[int, list[int]]) -> str:
    """Build a mock alignment response from (claim_index, citation_numbers) pairs."""
    return json.dumps(
        {
            "alignments": [
                {"claim_index": idx, "citation_numbers": numbers} for idx, numbers in pairs
            ]
        }
    )


def test_valid_citation_numbers_produce_doc_ids():
    aligner = _make_aligner()
    claims = [Claim(text="Paris is the capital of France.", span=(-1, -1))]
    with patch.object(aligner.client, "call", return_value=_response((0, [1]))):
        aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == ["doc-france"]


def test_span_set_from_single_candidate_sentence():
    aligner = _make_aligner()
    claims = [Claim(text="The Eiffel Tower is 330 metres high.", span=(-1, -1))]
    with patch.object(aligner.client, "call", return_value=_response((0, [2]))):
        aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    start, end = aligned[0].claim.span
    assert (start, end) != (-1, -1)
    assert TEXT[start:end].startswith("The Eiffel Tower")


def test_span_tiebreak_among_multiple_candidates():
    """
    When a citation number appears in more than one sentence, the span
    should be tie-broken toward the sentence that actually matches the
    claim's content, not just the first one carrying that marker.
    """
    text = (
        "Paris is the capital of France [1]. "
        "The Louvre is in Paris and is a famous museum [1]. "
        "The Eiffel Tower stands 330 m [2]."
    )
    aligner = _make_aligner()
    claims = [Claim(text="The Louvre is a famous museum in Paris.", span=(-1, -1))]
    with patch.object(aligner.client, "call", return_value=_response((0, [1]))):
        aligned = aligner.align(claims, text, CITATION_TO_DOC)
    # Citation is correct either way (both candidate sentences carry [1])...
    assert aligned[0].citation_doc_ids == ["doc-france"]
    # ...but the span should point at the Louvre sentence, not the France one.
    start, end = aligned[0].claim.span
    assert "Louvre" in text[start:end]


def test_empty_citation_numbers_stays_uncited():
    aligner = _make_aligner()
    claims = [Claim(text="Blockchain technology.", span=(-1, -1))]
    with patch.object(aligner.client, "call", return_value=_response((0, []))):
        aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []
    assert aligned[0].claim.span == (-1, -1)


def test_hallucinated_citation_number_is_dropped(caplog):
    aligner = _make_aligner()
    claims = [Claim(text="Paris is the capital of France.", span=(-1, -1))]
    with patch.object(aligner.client, "call", return_value=_response((0, [99]))):
        with caplog.at_level("WARNING"):
            aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []
    assert any("99" in rec.message for rec in caplog.records)


def test_incomplete_response_falls_back_gracefully_after_retries():
    """A response missing claim indices should retry, then degrade to
    empty citations rather than raising out of align()."""
    aligner = _make_aligner(max_retries=2)
    claims = [
        Claim(text="Paris is the capital of France.", span=(-1, -1)),
        Claim(text="The Eiffel Tower is 330 metres high.", span=(-1, -1)),
    ]
    # Only covers claim_index 0 — claim 1 is missing every retry.
    with patch.object(aligner.client, "call", return_value=_response((0, [1]))):
        aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []
    assert aligned[1].citation_doc_ids == []


def test_claim_with_real_span_bypasses_llm():
    """Claims that already carry a real span skip the LLM entirely (same
    defensive passthrough as JaccardCitationAligner)."""
    aligner = _make_aligner()
    claims = [Claim(text="Paris is the capital of France.", span=(0, 35))]
    with patch.object(aligner.client, "call") as mock_call:
        aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    mock_call.assert_not_called()
    assert aligned[0].citation_doc_ids == ["doc-france"]


# ── Regressions against confirmed real-data bugs (manual audit of
#    outputs/alce_eval/*.jsonl) ────────────────────────────────────────────
#
# Both scenarios below are cases where JaccardCitationAligner produced a
# confirmed-wrong citation_doc_ids; the LLM aligner is given the *correct*
# attribution (as a real judge would supply it) to confirm the plumbing
# preserves it — these test the aligner's mechanics, not model accuracy.


def test_qampari_list_answer_does_not_over_attribute_citations():
    """
    Real bug (qampari_gemma-4-e4b.jsonl, item 9): pysbd splits a whole
    comma-separated QAMPARI answer as one "sentence", so
    JaccardCitationAligner attributed the union of all 3 citations to every
    one of the 4 claims, regardless of which diamond each was about. The
    LLM aligner must keep each claim's citation to its own entity.
    """
    text = (
        "Pink Panther diamond [1], Darya-ye Noor diamond [1], "
        "DeYoung Red Diamond [2], Pink Star [3]."
    )
    citation_to_doc = {1: "doc-pink-panther", 2: "doc-deyoung", 3: "doc-pink-star"}
    claims = [
        Claim(text="The Pink Panther is a diamond.", span=(-1, -1)),
        Claim(text="The Darya-ye Noor is a diamond.", span=(-1, -1)),
        Claim(text="The DeYoung Red Diamond is a diamond.", span=(-1, -1)),
        Claim(text="The Pink Star is a diamond.", span=(-1, -1)),
    ]
    aligner = _make_aligner()
    response = _response((0, [1]), (1, [1]), (2, [2]), (3, [3]))
    with patch.object(aligner.client, "call", return_value=response):
        aligned = aligner.align(claims, text, citation_to_doc)

    assert aligned[0].citation_doc_ids == ["doc-pink-panther"]
    assert aligned[1].citation_doc_ids == ["doc-pink-panther"]
    assert aligned[2].citation_doc_ids == ["doc-deyoung"]
    assert aligned[3].citation_doc_ids == ["doc-pink-star"]


def test_im_coming_out_regression():
    """
    Real bug (asqa_llama-3.1-8b.jsonl, item 1): 7 of 12 claims (writer,
    producer, release date, album facts) were matched by
    JaccardCitationAligner to the uncited intro sentence instead of the
    cited, detail-dense second sentence two sentences later, losing their
    citation entirely. The LLM aligner (given the correct attribution)
    must land these on doc-song, not empty.
    """
    text = (
        'The song "I\'m Coming Out" was recorded by American singer Diana Ross. '
        "It was written and produced by Chic members Bernard Edwards and Nile "
        "Rodgers, and released in August 22, 1980 as the second single from "
        "Ross' self-titled tenth album \"Diana\" (1980) [1]."
    )
    citation_to_doc = {1: "doc-song"}
    claims = [
        Claim(text="Bernard Edwards wrote the song 'I'm Coming Out'.", span=(-1, -1)),
        Claim(text="Nile Rodgers produced the song 'I'm Coming Out'.", span=(-1, -1)),
        Claim(text="The song 'I'm Coming Out' was released on August 22, 1980.", span=(-1, -1)),
    ]
    aligner = _make_aligner()
    response = _response((0, [1]), (1, [1]), (2, [1]))
    with patch.object(aligner.client, "call", return_value=response):
        aligned = aligner.align(claims, text, citation_to_doc)

    for a in aligned:
        assert a.citation_doc_ids == ["doc-song"], a.claim.text
