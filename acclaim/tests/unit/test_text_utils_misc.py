"""
Unit tests for text_utils.py functions not covered by
test_text_utils_sentence_split.py: split_sentences_preserve_citations,
tokenize_text, jaccard_similarity.
"""

from acclaim.text_utils import (
    jaccard_similarity,
    split_sentences_preserve_citations,
    tokenize_text,
)


# ---------------------------------------------------------------------------
# split_sentences_preserve_citations
# ---------------------------------------------------------------------------


def test_preserve_citations_reattaches_leading_marker():
    text = "Paris is the capital of France. [1] The Eiffel Tower stands 330 m."
    sentences = split_sentences_preserve_citations(text)
    assert sentences == [
        "Paris is the capital of France. [1]",
        "The Eiffel Tower stands 330 m.",
    ]


def test_preserve_citations_reattaches_multiple_leading_markers():
    text = "Paris is the capital of France. [1][2] The Eiffel Tower stands 330 m."
    sentences = split_sentences_preserve_citations(text)
    assert sentences[0] == "Paris is the capital of France. [1] [2]"
    assert sentences[1] == "The Eiffel Tower stands 330 m."


def test_preserve_citations_no_leading_marker_unchanged():
    text = "Paris is the capital of France. The Eiffel Tower stands 330 m."
    sentences = split_sentences_preserve_citations(text)
    assert sentences == [
        "Paris is the capital of France.",
        "The Eiffel Tower stands 330 m.",
    ]


def test_preserve_citations_empty_text():
    assert split_sentences_preserve_citations("") == []


def test_preserve_citations_marker_as_first_sentence_kept_standalone():
    # No preceding sentence to attach to -> stays as its own entry.
    text = "[1] The Eiffel Tower stands 330 m."
    sentences = split_sentences_preserve_citations(text)
    assert sentences == ["[1] The Eiffel Tower stands 330 m."]


# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize_text("Paris, is great!") == {"paris", "is", "great"}


def test_tokenize_deduplicates_via_set():
    assert tokenize_text("the the cat cat") == {"the", "cat"}


def test_tokenize_empty_string():
    assert tokenize_text("") == set()


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------


def test_jaccard_similarity_identical_sets():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_similarity_disjoint_sets():
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


def test_jaccard_similarity_partial_overlap():
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == 1 / 3


def test_jaccard_similarity_both_empty_returns_zero():
    assert jaccard_similarity(set(), set()) == 0.0
