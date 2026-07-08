"""
Unit tests for the sentence claim extractor.
"""

import pytest

from acclaim.claims.sentence import SentenceClaimExtractor
from acclaim.data_models import Answer


def test_basic_extraction():
    text = "The Eiffel Tower is in Paris [1]. It was completed in 1889 [2]."
    extractor = SentenceClaimExtractor()
    claims = extractor.extract(Answer(text=text))

    assert len(claims) == 2
    assert "Eiffel Tower" in claims[0].text
    assert "[1]" not in claims[0].text
    assert "[2]" not in claims[1].text


def test_spans_cover_original_text():
    text = "Paris is a city [1]. France is a country [2]."
    extractor = SentenceClaimExtractor()
    claims = extractor.extract(Answer(text=text))

    for claim in claims:
        start, end = claim.span
        assert 0 <= start < end <= len(text)
        # The span in the original text must contain the clean claim words
        span_text = text[start:end]
        # At least some overlap with the clean text
        first_word = claim.text.split()[0]
        assert first_word in span_text


def test_empty_answer():
    extractor = SentenceClaimExtractor()
    claims = extractor.extract(Answer(text=""))
    assert claims == []


def test_single_sentence_no_citation():
    text = "The sky is blue."
    extractor = SentenceClaimExtractor()
    claims = extractor.extract(Answer(text=text))
    assert len(claims) == 1
    assert claims[0].text == "The sky is blue."
    assert claims[0].span == (0, len(text))


def test_citation_markers_stripped():
    text = "Quantum mechanics [1][2] is complex."
    extractor = SentenceClaimExtractor()
    claims = extractor.extract(Answer(text=text))
    assert len(claims) == 1
    assert "[1]" not in claims[0].text
    assert "[2]" not in claims[0].text
    assert "Quantum mechanics" in claims[0].text
