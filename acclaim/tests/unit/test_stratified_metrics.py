"""
Unit tests for RelevanceStratifiedMetric.

Example scenario
----------------
Question: "What are the main causes of type 2 diabetes?"

Answer (paraphrased, with citations):
  [1] Type 2 diabetes is primarily driven by insulin resistance.       → CORE
  [2] Obesity accounts for roughly 80% of cases.                       → CORE
  [3] Regular physical activity can significantly reduce the risk.     → COMPLEMENTARY
  [4] The disease was previously known as adult-onset diabetes.        → COMPLEMENTARY
  [5] Insulin was first isolated by Banting and Best in 1921.         → IRRELEVANT

CitationCorrectness is used as the base metric because it is pure
computation over SupportLabel values — no LLM calls needed.
"""

from __future__ import annotations

import threading

import pytest

from acclaim.data_models import (
    Claim,
    ClaimResult,
    RelevanceLabel,
    RelevanceResult,
    SupportLabel,
    SupportResult,
)
from acclaim.judges.relevance import RelevanceJudge
from acclaim.metrics.correctness import CitationCorrectness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUESTION = "What are the main causes of type 2 diabetes?"


def _make_cr(
    text: str,
    doc_ids: list[str],
    label: SupportLabel = SupportLabel.SUPPORTED,
) -> ClaimResult:
    return ClaimResult(
        claim=Claim(text=text, span=(-1, -1)),
        citation_doc_ids=doc_ids,
        support=SupportResult(label=label, confidence=0.9, reason="test"),
    )


def _relevance(label: RelevanceLabel) -> RelevanceResult:
    return RelevanceResult(label=label, confidence=0.95, reason="test reason here")


class _FixedRelevanceJudge(RelevanceJudge):
    """Stub judge that returns a predetermined label per claim text."""

    def __init__(self, mapping: dict[str, RelevanceLabel]) -> None:
        self._mapping = mapping

    def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
        label = self._mapping[claim.text]
        return _relevance(label)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Five claims from the diabetes example, each with a citation and a support label.
# Two CORE (both supported), two COMPLEMENTARY (one supported, one refuted),
# one IRRELEVANT (supported).
@pytest.fixture
def diabetes_claims() -> list[ClaimResult]:
    return [
        _make_cr("Insulin resistance causes type 2 diabetes", ["doc-1"], SupportLabel.SUPPORTED),
        _make_cr("Obesity accounts for 80% of cases", ["doc-2"], SupportLabel.SUPPORTED),
        _make_cr("Physical activity reduces the risk", ["doc-3"], SupportLabel.SUPPORTED),
        _make_cr("Previously called adult-onset diabetes", ["doc-4"], SupportLabel.REFUTED),
        _make_cr("Insulin isolated by Banting and Best in 1921", ["doc-5"], SupportLabel.SUPPORTED),
    ]


@pytest.fixture
def diabetes_judge() -> _FixedRelevanceJudge:
    return _FixedRelevanceJudge({
        "Insulin resistance causes type 2 diabetes": RelevanceLabel.CORE,
        "Obesity accounts for 80% of cases": RelevanceLabel.CORE,
        "Physical activity reduces the risk": RelevanceLabel.COMPLEMENTARY,
        "Previously called adult-onset diabetes": RelevanceLabel.COMPLEMENTARY,
        "Insulin isolated by Banting and Best in 1921": RelevanceLabel.IRRELEVANT,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRelevanceStratifiedMetric:
    """Tests for RelevanceStratifiedMetric wrapping CitationCorrectness."""

    def _make_metric(self, judge: RelevanceJudge):
        from acclaim.metrics.stratified import RelevanceStratifiedMetric
        return RelevanceStratifiedMetric(
            base_metric=CitationCorrectness(),
            relevance_judge=judge,
        )

    def test_output_keys_all_strata(self, diabetes_claims, diabetes_judge):
        """All three strata present → output contains the un-stratified base plus three stratum keys."""
        metric = self._make_metric(diabetes_judge)
        result = metric.compute_stratified(diabetes_claims, QUESTION)
        assert set(result.keys()) == {
            "citation_correctness",
            "citation_correctness_core",
            "citation_correctness_complementary",
            "citation_correctness_irrelevant",
        }

    def test_core_correctness(self, diabetes_claims, diabetes_judge):
        """CORE stratum: both claims are SUPPORTED → correctness = 1.0."""
        metric = self._make_metric(diabetes_judge)
        result = metric.compute_stratified(diabetes_claims, QUESTION)
        assert result["citation_correctness_core"] == pytest.approx(1.0)

    def test_complementary_correctness(self, diabetes_claims, diabetes_judge):
        """COMPLEMENTARY stratum: 1 SUPPORTED + 1 REFUTED → correctness = 0.5."""
        metric = self._make_metric(diabetes_judge)
        result = metric.compute_stratified(diabetes_claims, QUESTION)
        assert result["citation_correctness_complementary"] == pytest.approx(0.5)

    def test_irrelevant_correctness(self, diabetes_claims, diabetes_judge):
        """IRRELEVANT stratum: 1 SUPPORTED → correctness = 1.0."""
        metric = self._make_metric(diabetes_judge)
        result = metric.compute_stratified(diabetes_claims, QUESTION)
        assert result["citation_correctness_irrelevant"] == pytest.approx(1.0)

    def test_empty_stratum_omitted(self):
        """A stratum with no claims produces no key in the output."""
        claims = [
            _make_cr("Insulin resistance causes type 2 diabetes", ["doc-1"], SupportLabel.SUPPORTED),
            _make_cr("Obesity accounts for 80% of cases", ["doc-2"], SupportLabel.REFUTED),
        ]
        judge = _FixedRelevanceJudge({
            "Insulin resistance causes type 2 diabetes": RelevanceLabel.CORE,
            "Obesity accounts for 80% of cases": RelevanceLabel.CORE,
        })
        metric = self._make_metric(judge)
        result = metric.compute_stratified(claims, QUESTION)
        assert "citation_correctness_complementary" not in result
        assert "citation_correctness_irrelevant" not in result
        assert "citation_correctness_core" in result

    def test_judge_called_once_per_claim(self, diabetes_claims):
        """The relevance judge is called exactly once per claim."""
        calls: list[tuple[str, str]] = []

        class _TrackingJudge(RelevanceJudge):
            def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
                calls.append((claim.text, question))
                return _relevance(RelevanceLabel.CORE)

        metric = self._make_metric(_TrackingJudge())
        metric.compute_stratified(diabetes_claims, QUESTION)
        assert len(calls) == len(diabetes_claims)

    def test_judge_receives_correct_question(self, diabetes_claims):
        """The relevance judge receives the question passed to compute_stratified."""
        received_questions: list[str] = []

        class _TrackingJudge(RelevanceJudge):
            def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
                received_questions.append(question)
                return _relevance(RelevanceLabel.CORE)

        metric = self._make_metric(_TrackingJudge())
        metric.compute_stratified(diabetes_claims, QUESTION)
        assert all(q == QUESTION for q in received_questions)

    def test_all_claims_same_stratum(self):
        """When all claims land in one stratum, only that key is present."""
        claims = [
            _make_cr("Insulin resistance causes type 2 diabetes", ["doc-1"], SupportLabel.SUPPORTED),
            _make_cr("Obesity accounts for 80% of cases", ["doc-2"], SupportLabel.SUPPORTED),
        ]
        judge = _FixedRelevanceJudge({
            "Insulin resistance causes type 2 diabetes": RelevanceLabel.CORE,
            "Obesity accounts for 80% of cases": RelevanceLabel.CORE,
        })
        metric = self._make_metric(judge)
        result = metric.compute_stratified(claims, QUESTION)
        assert set(result.keys()) == {"citation_correctness", "citation_correctness_core"}
        assert result["citation_correctness_core"] == pytest.approx(1.0)
        assert result["citation_correctness"] == pytest.approx(1.0)

    def test_empty_claim_list(self):
        """Empty input still emits the un-stratified base (0.0) — no stratum keys."""
        metric = self._make_metric(_FixedRelevanceJudge({}))
        result = metric.compute_stratified([], QUESTION)
        assert result == {"citation_correctness": 0.0}

    def test_compute_fallback_without_question(self, diabetes_claims, diabetes_judge):
        """compute() without a question returns 0.0 (StratifiedMetric fallback)."""
        metric = self._make_metric(diabetes_judge)
        assert metric.compute(diabetes_claims) == pytest.approx(0.0)

    def test_name(self, diabetes_judge):
        """Metric name follows the {base_metric_name}_stratified convention."""
        metric = self._make_metric(diabetes_judge)
        assert metric.name == "citation_correctness_stratified"

    def test_max_workers_calls_judge_once_per_claim_and_attributes_correctly(
        self, diabetes_claims, diabetes_judge
    ):
        """With max_workers > 1, the judge may run on multiple threads, but is
        still called exactly once per claim, and results stay attributed to
        the correct claim despite concurrent dispatch."""
        from acclaim.metrics.stratified import RelevanceStratifiedMetric

        calls: list[str] = []
        lock = threading.Lock()

        class _TrackingJudge(RelevanceJudge):
            def evaluate(self, claim: Claim, question: str) -> RelevanceResult:
                with lock:
                    calls.append(claim.text)
                return diabetes_judge.evaluate(claim, question)

        metric = RelevanceStratifiedMetric(
            base_metric=CitationCorrectness(),
            relevance_judge=_TrackingJudge(),
            max_workers=4,
        )
        result = metric.compute_stratified(diabetes_claims, QUESTION)

        assert len(calls) == len(diabetes_claims)
        assert result["citation_correctness_core"] == pytest.approx(1.0)
        assert result["citation_correctness_complementary"] == pytest.approx(0.5)
        assert result["citation_correctness_irrelevant"] == pytest.approx(1.0)
