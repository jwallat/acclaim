"""Unit tests for CVCPMetric and split_sentences_preserve_citations."""

import math

import pytest

from acclaim.metrics.cvcp import CVCPMetric
from acclaim.text_utils import split_sentences_preserve_citations


# ---------------------------------------------------------------------------
# split_sentences_preserve_citations
# ---------------------------------------------------------------------------


class TestSplitSentencesPreserveCitations:
    def test_no_dangling_citations(self):
        text = "Paris is the capital of France [1]. The Eiffel Tower was built in 1889 [2]."
        result = split_sentences_preserve_citations(text)
        assert result == [
            "Paris is the capital of France [1].",
            "The Eiffel Tower was built in 1889 [2].",
        ]

    def test_dangling_single_citation_stitched_to_previous(self):
        # Splitter breaks "France. [1] The" into ["France.", "[1] The ..."]
        # The [1] belongs to "France." and must be stitched back.
        text = "Paris is the capital of France. [1] The Eiffel Tower was built in 1889."
        result = split_sentences_preserve_citations(text)
        assert result[0] == "Paris is the capital of France. [1]"
        assert result[1] == "The Eiffel Tower was built in 1889."

    def test_dangling_multiple_citations_stitched(self):
        text = "Paris is in France. [1][2] Next sentence."
        result = split_sentences_preserve_citations(text)
        assert result[0] == "Paris is in France. [1] [2]"
        assert result[1] == "Next sentence."

    def test_first_sentence_starting_with_citation_not_stitched(self):
        # No previous sentence to stitch to — stays as-is.
        text = "[1] Paris is the capital of France."
        result = split_sentences_preserve_citations(text)
        assert len(result) == 1
        assert result[0] == "[1] Paris is the capital of France."

    def test_empty_string(self):
        assert split_sentences_preserve_citations("") == []

    def test_no_citations(self):
        text = "Paris is the capital of France. The Eiffel Tower was built in 1889."
        result = split_sentences_preserve_citations(text)
        assert result == [
            "Paris is the capital of France.",
            "The Eiffel Tower was built in 1889.",
        ]

    def test_dangling_citation_with_trailing_text_becomes_new_sentence(self):
        # "[1] More text." — the [1] goes to previous, "More text." becomes new sentence.
        text = "First sentence. [1] More text."
        result = split_sentences_preserve_citations(text)
        assert result[0] == "First sentence. [1]"
        assert result[1] == "More text."


# ---------------------------------------------------------------------------
# CVCPMetric
# ---------------------------------------------------------------------------


class TestCVCPMetric:
    metric = CVCPMetric()

    def test_name(self):
        assert self.metric.name == "cvcp"

    def test_none_answer_text_returns_zero(self):
        assert self.metric.compute([], answer_text=None) == pytest.approx(0.0)

    def test_empty_answer_returns_zero(self):
        assert self.metric.compute([], answer_text="") == pytest.approx(0.0)

    def test_no_citations_returns_zero(self):
        assert self.metric.compute(
            [], answer_text="Paris is the capital of France. No citations here."
        ) == pytest.approx(0.0)

    def test_single_citation_sigma_is_zero(self):
        # One marker per sentence → σ=0 → CV=0 → CVCP=0.
        assert self.metric.compute(
            [], answer_text="Paris is the capital of France [1]."
        ) == pytest.approx(0.0)

    def test_two_citations_spread_across_sentence(self):
        # "Hello [1] world [2]." len=20
        # positions: 6/20=0.3, 16/20=0.8
        # mu=0.55, variance=0.0625, sigma=0.25, CV=0.25/0.55=5/11
        sentence = "Hello [1] world [2]."
        assert len(sentence) == 20
        expected = 0.25 / 0.55
        assert self.metric.compute([], answer_text=sentence) == pytest.approx(
            expected, rel=1e-6
        )

    def test_uncited_sentences_excluded_from_average(self):
        # Sentence 1: "Hello [1] world [2]." → CV = 5/11
        # Sentence 2: "No citation here."    → skipped
        # Sentence 3: "Foo [3]."             → CV = 0.0
        # CVCP = (5/11 + 0.0) / 2
        answer = "Hello [1] world [2]. No citation here. Foo [3]."
        expected = (0.25 / 0.55 + 0.0) / 2
        assert self.metric.compute([], answer_text=answer) == pytest.approx(
            expected, rel=1e-6
        )

    def test_all_citations_at_position_zero_skipped(self):
        # Sentence starts with [1] (position 0) → mu=0 → skip → returns 0.0.
        assert self.metric.compute(
            [], answer_text="[1] Paris is the capital of France."
        ) == pytest.approx(0.0)

    def test_claim_results_ignored(self):
        # The metric derives everything from answer_text; claim_results has no effect.
        from acclaim.data_models import Claim, ClaimResult, SupportLabel, SupportResult

        cr = ClaimResult(
            claim=Claim(text="Irrelevant", span=(-1, -1)),
            citation_doc_ids=["doc1"],
            support=SupportResult(label=SupportLabel.SUPPORTED, confidence=1.0, reason=""),
        )
        score_empty = self.metric.compute([], answer_text="Hello [1] world [2].")
        score_with_cr = self.metric.compute([cr], answer_text="Hello [1] world [2].")
        assert score_empty == pytest.approx(score_with_cr)

    def test_multiple_sentences_all_with_citations(self):
        # Two identical sentences → CVCP equals the per-sentence CV.
        sentence = "Hello [1] world [2]."
        answer = f"{sentence} {sentence}"
        expected_single = 0.25 / 0.55
        assert self.metric.compute([], answer_text=answer) == pytest.approx(
            expected_single, rel=1e-6
        )
