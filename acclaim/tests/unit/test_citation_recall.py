"""
Unit tests for CitationRecallMetric, CitationPrecisionMetric,
CitationF1Metric, LiteLLMRecallJudge, and LiteLLMPrecisionJudge.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from acclaim.data_models import (
    Claim,
    ClaimResult,
    Document,
    SupportLabel,
    SupportResult,
)
from acclaim.judges.citation_recall import (
    LiteLLMRecallJudge,
    RecallJudge,
    RecallLabel,
    RecallResult,
)
from acclaim.metrics.base import Metric
from acclaim.metrics.citation_recall import CitationRecallMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_doc(doc_id: str, content: str = "Some text.") -> Document:
    return Document(doc_id=doc_id, text=content)


def _recall_result(label: RecallLabel) -> RecallResult:
    return RecallResult(label=label, score=label.score, reason="test reason here")


class _FixedRecallJudge(RecallJudge):
    """Stub judge that returns a fixed sequence of RecallResults."""

    def __init__(self, results: list[RecallResult]) -> None:
        self._results = iter(results)

    def recall(self, claim: Claim, docs: list[Document]) -> RecallResult:
        return next(self._results)


# ---------------------------------------------------------------------------
# RecallLabel.score
# ---------------------------------------------------------------------------


class TestRecallLabel:
    def test_full_score(self):
        assert RecallLabel.FULL.score == 1.0

    def test_partial_score(self):
        assert RecallLabel.PARTIAL.score == 0.5

    def test_none_score(self):
        assert RecallLabel.NONE.score == 0.0


# ---------------------------------------------------------------------------
# CitationRecallMetric
# ---------------------------------------------------------------------------


class TestCitationRecallMetric:
    docs = [_make_doc("doc-1"), _make_doc("doc-2"), _make_doc("doc-3")]

    def _metric(self, results: list[RecallResult]) -> CitationRecallMetric:
        return CitationRecallMetric(judge=_FixedRecallJudge(results))

    # ------ basic cases ------

    def test_empty_claim_list_returns_zero(self):
        metric = self._metric([])
        assert metric.compute([]) == 0.0

    def test_name(self):
        metric = self._metric([])
        assert metric.name == "citation_recall"

    def test_all_fully_supported(self):
        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
        ]
        judge_results = [_recall_result(RecallLabel.FULL)] * 2
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(1.0)

    def test_all_partially_supported(self):
        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
        ]
        judge_results = [_recall_result(RecallLabel.PARTIAL)] * 2
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(0.5)

    def test_all_none(self):
        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
        ]
        judge_results = [_recall_result(RecallLabel.NONE)] * 2
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(0.0)

    def test_mixed_labels(self):
        # FULL=1.0, PARTIAL=0.5, NONE=0.0 → mean = 0.5
        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
            _make_cr("Claim C", ["doc-3"]),
        ]
        judge_results = [
            _recall_result(RecallLabel.FULL),
            _recall_result(RecallLabel.PARTIAL),
            _recall_result(RecallLabel.NONE),
        ]
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        assert score == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    # ------ uncited claims ------

    def test_uncited_claims_score_zero(self):
        claim_results = [
            _make_cr("Claim A", []),
            _make_cr("Claim B", []),
        ]
        score = self._metric([]).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(0.0)

    def test_mixed_cited_and_uncited(self):
        # Cited: FULL (1.0); Uncited: 0.0 → mean = 0.5
        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", []),
        ]
        judge_results = [_recall_result(RecallLabel.FULL)]
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(0.5)

    # ------ document resolution ------

    def test_documents_none_scores_zero_for_cited(self):
        """When documents=None, cited claims can't be resolved → score 0."""
        claim_results = [_make_cr("Claim A", ["doc-1"])]
        score = self._metric([]).compute(claim_results, documents=None)
        assert score == pytest.approx(0.0)

    def test_missing_doc_id_scores_zero(self):
        """Cited doc ID not in documents → score 0."""
        claim_results = [_make_cr("Claim A", ["doc-99"])]
        score = self._metric([]).compute(claim_results, documents=self.docs)
        assert score == pytest.approx(0.0)

    def test_partial_doc_resolution(self):
        """Only some cited IDs found — judge is still called with resolved docs."""
        claim_results = [_make_cr("Claim A", ["doc-1", "doc-99"])]
        judge_results = [_recall_result(RecallLabel.FULL)]
        score = self._metric(judge_results).compute(claim_results, documents=self.docs)
        # doc-1 resolves, doc-99 doesn't — judge is called with [doc-1] → FULL
        assert score == pytest.approx(1.0)

    def test_max_workers_attributes_scores_to_correct_claim(self):
        """With max_workers > 1, calls run concurrently but each claim still
        gets the correct score — uses a content-keyed judge (not an
        order-dependent sequence) since concurrent dispatch doesn't guarantee
        call order."""
        calls: list[str] = []
        lock = threading.Lock()

        class _KeyedRecallJudge(RecallJudge):
            def recall(self, claim, docs):
                with lock:
                    calls.append(claim.text)
                label = RecallLabel.FULL if claim.text == "Claim B" else RecallLabel.NONE
                return _recall_result(label)

        claim_results = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
            _make_cr("Claim C", ["doc-3"]),
        ]
        metric = CitationRecallMetric(judge=_KeyedRecallJudge(), max_workers=4)
        score = metric.compute(claim_results, documents=self.docs)

        assert sorted(calls) == ["Claim A", "Claim B", "Claim C"]
        assert score == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# LiteLLMRecallJudge
# ---------------------------------------------------------------------------


class TestLiteLLMRecallJudge:
    def test_empty_docs_returns_none_score(self):
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        claim = Claim(text="Some claim text.", span=(-1, -1))
        result = judge.recall(claim, [])
        assert result.label == RecallLabel.NONE
        assert result.score == 0.0

    def test_parse_full_response(self):
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        raw = '{"label": "FULL", "reason": "The evidence fully supports the claim."}'
        result = judge._parse_response(raw)
        assert result.label == RecallLabel.FULL
        assert result.score == 1.0

    def test_parse_partial_response(self):
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        raw = '{"label": "PARTIAL", "reason": "Only part is supported here."}'
        result = judge._parse_response(raw)
        assert result.label == RecallLabel.PARTIAL
        assert result.score == 0.5

    def test_parse_none_response(self):
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        raw = '{"label": "NONE", "reason": "The evidence does not support the claim at all."}'
        result = judge._parse_response(raw)
        assert result.label == RecallLabel.NONE
        assert result.score == 0.0

    def test_llm_failure_returns_none_score(self):
        """Any exception from the LLM client should return score 0."""
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        claim = Claim(text="Some claim text here.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Irrelevant text about something else.")
        with patch.object(
            judge.client, "call_with_retry", side_effect=RuntimeError("API error")
        ):
            result = judge.recall(claim, [doc])
        assert result.label == RecallLabel.NONE
        assert result.score == 0.0

    def test_successful_llm_call(self):
        """Successful LLM call is parsed and returned correctly."""
        judge = LiteLLMRecallJudge(model="gpt-4o-mini")
        claim = Claim(text="Paris is the capital of France.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Paris is the capital city of France.")
        expected = RecallResult(
            label=RecallLabel.FULL,
            score=1.0,
            reason="Evidence fully supports the claim.",
        )
        with patch.object(judge.client, "call_with_retry", return_value=expected):
            result = judge.recall(claim, [doc])
        assert result.label == RecallLabel.FULL
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# CitationPrecisionMetric (stub)
# ---------------------------------------------------------------------------


class TestCitationPrecisionMetric:
    def test_name(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _DummyPrecisionJudge(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT, score=1.0, reason="ok"
                )

        assert (
            CitationPrecisionMetric(judge=_DummyPrecisionJudge()).name
            == "citation_precision"
        )

    def test_all_relevant(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _Always1(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT, score=1.0, reason="ok"
                )

        crs = [
            _make_cr("Claim A", ["doc-1"]),
            _make_cr("Claim B", ["doc-2"]),
        ]
        docs = [_make_doc("doc-1"), _make_doc("doc-2")]
        score = CitationPrecisionMetric(judge=_Always1()).compute(crs, documents=docs)
        assert score == pytest.approx(1.0)

    def test_all_unrelevant(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _Always0(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.UNRELEVANT, score=0.0, reason="nope"
                )

        crs = [_make_cr("Claim A", ["doc-1"])]
        docs = [_make_doc("doc-1")]
        score = CitationPrecisionMetric(judge=_Always0()).compute(crs, documents=docs)
        assert score == pytest.approx(0.0)

    def test_multiple_citations_per_claim(self):
        """Precision is per-citation: 3 citations from one claim produce 3 scores."""
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )
        from itertools import cycle

        scores_seq = [1.0, 0.0, 1.0]  # 2 relevant out of 3
        it = iter(scores_seq)

        class _Seq(PrecisionJudge):
            def judge(self, claim, doc):
                v = next(it)
                lbl = PrecisionLabel.RELEVANT if v == 1.0 else PrecisionLabel.UNRELEVANT
                return PrecisionResult(label=lbl, score=v, reason="test")

        crs = [_make_cr("Claim A", ["doc-1", "doc-2", "doc-3"])]
        docs = [_make_doc("doc-1"), _make_doc("doc-2"), _make_doc("doc-3")]
        score = CitationPrecisionMetric(judge=_Seq()).compute(crs, documents=docs)
        assert score == pytest.approx(2 / 3)

    def test_no_citations_returns_zero(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _Dummy(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT, score=1.0, reason="ok"
                )

        crs = [_make_cr("Claim A", [])]
        score = CitationPrecisionMetric(judge=_Dummy()).compute(crs, documents=[])
        assert score == pytest.approx(0.0)

    def test_empty_claims_returns_zero(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _Dummy(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT, score=1.0, reason="ok"
                )

        assert CitationPrecisionMetric(judge=_Dummy()).compute([]) == pytest.approx(0.0)

    def test_missing_doc_scores_zero(self):
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        class _Dummy(PrecisionJudge):
            def judge(self, claim, doc):
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT, score=1.0, reason="ok"
                )

        crs = [_make_cr("Claim A", ["doc-99"])]
        # doc-99 is not in documents list
        score = CitationPrecisionMetric(judge=_Dummy()).compute(
            crs, documents=[_make_doc("doc-1")]
        )
        assert score == pytest.approx(0.0)

    def test_max_workers_attributes_scores_to_correct_pair(self):
        """With max_workers > 1, calls run concurrently but each (claim, doc)
        pair is still scored correctly — uses a content-keyed judge (not an
        order-dependent sequence) since concurrent dispatch doesn't guarantee
        call order."""
        from acclaim.metrics.citation_precision import CitationPrecisionMetric
        from acclaim.judges.citation_precision import (
            PrecisionJudge,
            PrecisionResult,
            PrecisionLabel,
        )

        calls: list[str] = []
        lock = threading.Lock()

        class _KeyedJudge(PrecisionJudge):
            def judge(self, claim, doc):
                with lock:
                    calls.append(doc.doc_id)
                relevant = doc.doc_id == "doc-2"
                return PrecisionResult(
                    label=PrecisionLabel.RELEVANT if relevant else PrecisionLabel.UNRELEVANT,
                    score=1.0 if relevant else 0.0,
                    reason="keyed",
                )

        crs = [_make_cr("Claim A", ["doc-1", "doc-2", "doc-3"])]
        docs = [_make_doc("doc-1"), _make_doc("doc-2"), _make_doc("doc-3")]
        score = CitationPrecisionMetric(judge=_KeyedJudge(), max_workers=4).compute(
            crs, documents=docs
        )
        assert sorted(calls) == ["doc-1", "doc-2", "doc-3"]
        assert score == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# CitationF1Metric
# ---------------------------------------------------------------------------


class _FixedMetric(Metric):
    """Stub metric that always returns a fixed value."""

    def __init__(self, val: float, metric_name: str) -> None:
        self._val = val
        self._name = metric_name

    @property
    def name(self) -> str:
        return self._name

    def compute(
        self,
        claim_results,
        valid_doc_ids=None,
        documents=None,
        answer_text=None,
        question=None,
    ) -> float:
        return self._val


class TestCitationF1Metric:
    def _f1(self, p: float, r: float) -> float:
        from acclaim.metrics.citation_f1 import CitationF1Metric

        return CitationF1Metric(
            precision=_FixedMetric(p, "citation_precision"),
            recall=_FixedMetric(r, "citation_recall"),
        ).compute([])

    def test_name(self):
        from acclaim.metrics.citation_f1 import CitationF1Metric

        m = CitationF1Metric(
            precision=_FixedMetric(0.0, "citation_precision"),
            recall=_FixedMetric(0.0, "citation_recall"),
        )
        assert m.name == "citation_f1"

    def test_perfect_score(self):
        assert self._f1(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert self._f1(0.0, 1.0) == pytest.approx(0.0)

    def test_zero_recall(self):
        assert self._f1(1.0, 0.0) == pytest.approx(0.0)

    def test_both_zero(self):
        assert self._f1(0.0, 0.0) == pytest.approx(0.0)

    def test_harmonic_mean(self):
        # F1(0.6, 0.4) = 2*0.6*0.4 / (0.6+0.4) = 0.48
        assert self._f1(0.6, 0.4) == pytest.approx(0.48)

    def test_equal_p_r_equals_p(self):
        # When P == R, F1 == P == R
        assert self._f1(0.7, 0.7) == pytest.approx(0.7)

    def test_args_forwarded(self):
        """valid_doc_ids and documents are forwarded to component metrics."""
        from acclaim.metrics.citation_f1 import CitationF1Metric

        received: dict = {}

        class _Capturing(Metric):
            def __init__(self, val: float, n: str) -> None:
                self._val = val
                self._n = n

            @property
            def name(self) -> str:
                return self._n

            def compute(self, cr, valid_doc_ids=None, documents=None, answer_text=None, question=None) -> float:
                received[self._n] = (valid_doc_ids, documents)
                return self._val

        docs = [Document(doc_id="d1", text="text")]
        valid = ["d1"]
        CitationF1Metric(
            precision=_Capturing(0.5, "citation_precision"),
            recall=_Capturing(0.5, "citation_recall"),
        ).compute([], valid_doc_ids=valid, documents=docs)

        assert received["citation_precision"] == (valid, docs)
        assert received["citation_recall"] == (valid, docs)


# ---------------------------------------------------------------------------
# LiteLLMPrecisionJudge
# ---------------------------------------------------------------------------


class TestLiteLLMPrecisionJudge:
    def _judge(self) -> object:
        from acclaim.judges.citation_precision import LiteLLMPrecisionJudge

        return LiteLLMPrecisionJudge(model="gpt-4o-mini")

    def test_parse_relevant(self):
        judge = self._judge()
        raw = "Rating: [[Relevant]] Analysis: The snippet directly supports the statement."
        result = judge._parse_response(raw)
        from acclaim.judges.citation_precision import PrecisionLabel

        assert result.label == PrecisionLabel.RELEVANT
        assert result.score == 1.0
        assert "snippet" in result.reason

    def test_parse_unrelevant(self):
        judge = self._judge()
        raw = "Rating: [[Unrelevant]] Analysis: The snippet is completely off-topic."
        result = judge._parse_response(raw)
        from acclaim.judges.citation_precision import PrecisionLabel

        assert result.label == PrecisionLabel.UNRELEVANT
        assert result.score == 0.0

    def test_parse_case_insensitive(self):
        """[[relevant]] and [[RELEVANT]] should both parse correctly."""
        judge = self._judge()
        from acclaim.judges.citation_precision import PrecisionLabel

        assert (
            judge._parse_response("Rating: [[relevant]] ok").label
            == PrecisionLabel.RELEVANT
        )
        assert (
            judge._parse_response("Rating: [[UNRELEVANT]] ok").label
            == PrecisionLabel.UNRELEVANT
        )

    def test_parse_missing_label_raises(self):
        judge = self._judge()
        with pytest.raises(ValueError, match="Could not find"):
            judge._parse_response("I think it is somewhat related.")

    def test_llm_failure_returns_unrelevant(self):
        from acclaim.judges.citation_precision import (
            LiteLLMPrecisionJudge,
            PrecisionLabel,
        )

        judge = LiteLLMPrecisionJudge(model="gpt-4o-mini")
        claim = Claim(text="Paris is the capital of France.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Some unrelated text.")
        with patch.object(
            judge.client, "call_with_retry", side_effect=RuntimeError("API error")
        ):
            result = judge.judge(claim, doc)
        assert result.label == PrecisionLabel.UNRELEVANT
        assert result.score == 0.0

    def test_successful_llm_call(self):
        from acclaim.judges.citation_precision import (
            LiteLLMPrecisionJudge,
            PrecisionLabel,
            PrecisionResult,
        )

        judge = LiteLLMPrecisionJudge(model="gpt-4o-mini")
        claim = Claim(text="Paris is the capital of France.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Paris is the capital city of France.")
        expected = PrecisionResult(
            label=PrecisionLabel.RELEVANT, score=1.0, reason="Directly supported."
        )
        with patch.object(judge.client, "call_with_retry", return_value=expected):
            result = judge.judge(claim, doc)
        assert result.label == PrecisionLabel.RELEVANT
        assert result.score == 1.0

    def test_user_message_includes_statement_and_snippet(self):
        from acclaim.judges.citation_precision import LiteLLMPrecisionJudge

        judge = LiteLLMPrecisionJudge(model="gpt-4o-mini")
        claim = Claim(text="The sky is blue.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Blue is the color of a clear sky.")
        msg = judge._build_user_message(claim, doc)
        assert "<statement>" in msg and "The sky is blue." in msg
        assert "<snippet>" in msg and "Blue is the color" in msg
        assert "<question>" not in msg  # empty query omitted

    def test_user_message_includes_question_when_provided(self):
        from acclaim.judges.citation_precision import LiteLLMPrecisionJudge

        judge = LiteLLMPrecisionJudge(model="gpt-4o-mini")
        claim = Claim(text="The sky is blue.", span=(-1, -1))
        doc = Document(doc_id="d1", text="Blue is the color of a clear sky.")
        msg = judge._build_user_message(claim, doc, query="What color is the sky?")
        assert "<question>" in msg and "What color is the sky?" in msg
