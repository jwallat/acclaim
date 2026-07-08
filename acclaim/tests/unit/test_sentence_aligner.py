"""
Unit tests for the SentenceCitationAligner.
"""

import pytest

from acclaim.alignment.sentence import SentenceCitationAligner
from acclaim.data_models import Claim


def _make_aligner() -> SentenceCitationAligner:
    return SentenceCitationAligner()


# answer text used across multiple tests
TEXT = "Paris is the capital of France [1]. The Eiffel Tower stands 330 m [2][3]."

CITATION_TO_DOC = {1: "doc-france", 2: "doc-eiffel", 3: "doc-height"}


def test_sentence_gets_its_citations():
    aligner = _make_aligner()
    # Manually craft spans matching TEXT above
    claims = [
        Claim(text="Paris is the capital of France.", span=(0, 36)),
        Claim(text="The Eiffel Tower stands 330 m.", span=(38, 73)),
    ]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)

    assert aligned[0].citation_doc_ids == ["doc-france"]
    assert set(aligned[1].citation_doc_ids) == {"doc-eiffel", "doc-height"}


def test_no_citation_markers_in_span():
    aligner = _make_aligner()
    claims = [Claim(text="Something uncited.", span=(0, 18))]
    TEXT2 = "Something uncited. Cited sentence [1]."
    aligned = aligner.align(claims, TEXT2, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []


def test_unknown_citation_number_ignored():
    aligner = _make_aligner()
    # Citation [9] is not in citation_to_doc
    text = "Some claim [9]."
    claims = [Claim(text="Some claim.", span=(0, len(text)))]
    aligned = aligner.align(claims, text, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []


def test_span_minus_one_returns_empty():
    aligner = _make_aligner()
    claims = [Claim(text="Atomic claim without span.", span=(-1, -1))]
    aligned = aligner.align(claims, TEXT, CITATION_TO_DOC)
    assert aligned[0].citation_doc_ids == []


def test_comma_separated_markers():
    aligner = _make_aligner()
    text = "Brigitte is 25 years his senior [1, 3]."
    claims = [Claim(text="Brigitte is 25 years his senior.", span=(0, len(text)))]
    aligned = aligner.align(claims, text, CITATION_TO_DOC)
    assert set(aligned[0].citation_doc_ids) == {"doc-france", "doc-height"}


def test_comma_separated_markers_no_space():
    aligner = _make_aligner()
    text = "Brigitte is 25 years his senior [1,3]."
    claims = [Claim(text="Brigitte is 25 years his senior.", span=(0, len(text)))]
    aligned = aligner.align(claims, text, CITATION_TO_DOC)
    assert set(aligned[0].citation_doc_ids) == {"doc-france", "doc-height"}
