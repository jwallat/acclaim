"""
Unit tests for the evaluate() orchestration pipeline (all LLM calls mocked
via a fake EvidenceJudge / explicit overrides — no network calls).
"""

import pytest

from acclaim.claims.atomic import AtomicClaimExtractor
from acclaim.claims.sentence import SentenceClaimExtractor
from acclaim.config.load import EvalConfig, JudgeConfig
from acclaim.data_models import Claim, Document, SupportLabel, SupportResult
from acclaim.evaluate import (
    _build_extractor,
    _build_judge,
    _build_metric_overrides,
    evaluate,
)
from acclaim.judges.base import EvidenceJudge
from acclaim.judges.litellm import LiteLLMJudge
from acclaim.metrics.citation_f1 import CitationF1Metric
from acclaim.metrics.citation_precision import CitationPrecisionMetric
from acclaim.metrics.citation_recall import CitationRecallMetric


class _AlwaysSupportedJudge(EvidenceJudge):
    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
        return SupportResult(label=SupportLabel.SUPPORTED, confidence=1.0, reason="test")


def test_evaluate_happy_path_with_explicit_judge():
    documents = [
        Document(doc_id="doc1", text="Paris is the capital of France."),
        Document(doc_id="doc2", text="The Eiffel Tower was completed in 1889."),
    ]
    answer = "Paris is the capital of France [1]. The Eiffel Tower was built in 1799 [2]."
    cfg = EvalConfig(metrics=["coverage", "hallucination_rate"])

    result = evaluate(answer, documents, config=cfg, judge=_AlwaysSupportedJudge())

    assert len(result.claims) == 2
    assert result.metrics["coverage"] == pytest.approx(1.0)
    assert result.metrics["hallucination_rate"] == pytest.approx(0.0)
    assert result.answer == answer
    assert result.documents == documents
    assert result.metadata is not None
    assert "claims" in result.steps
    assert "aligned_claims" in result.steps
    assert "claim_results" in result.steps


def test_evaluate_defaults_citation_to_doc_by_position():
    documents = [Document(doc_id="doc-a", text="A."), Document(doc_id="doc-b", text="B.")]
    answer = "First fact [1]. Second fact [2]."
    cfg = EvalConfig(metrics=["coverage"])

    result = evaluate(answer, documents, config=cfg, judge=_AlwaysSupportedJudge())

    assert result.steps["citation_to_doc"] == {1: "doc-a", 2: "doc-b"}


def test_evaluate_respects_explicit_citation_to_doc():
    documents = [Document(doc_id="doc_paris", text="Paris is the capital of France.")]
    answer = "Paris is the capital of France [1]."
    cfg = EvalConfig(metrics=["coverage"])

    result = evaluate(
        answer,
        documents,
        config=cfg,
        judge=_AlwaysSupportedJudge(),
        citation_to_doc={1: "doc_paris"},
    )

    assert result.claims[0].citation_doc_ids == ["doc_paris"]


def test_evaluate_no_metadata_when_disabled():
    documents = [Document(doc_id="doc1", text="A.")]
    answer = "A fact [1]."
    cfg = EvalConfig(metrics=["coverage"])

    result = evaluate(
        answer,
        documents,
        config=cfg,
        judge=_AlwaysSupportedJudge(),
        _include_metadata=False,
    )

    assert result.metadata is None


# ---------------------------------------------------------------------------
# _build_extractor
# ---------------------------------------------------------------------------


def test_build_extractor_sentence():
    cfg = EvalConfig(claim_extractor="sentence")
    assert isinstance(_build_extractor(cfg), SentenceClaimExtractor)


def test_build_extractor_atomic_inherits_judge_connection_fields():
    cfg = EvalConfig(
        claim_extractor="atomic",
        judge=JudgeConfig(model="gpt-4o", api_key="judge-key", temperature=0.3),
    )
    extractor = _build_extractor(cfg)
    assert isinstance(extractor, AtomicClaimExtractor)
    assert extractor.client.model == "gpt-4o"
    assert extractor.client.api_key == "judge-key"
    assert extractor.client.temperature == 0.3


def test_build_extractor_unknown_key_raises():
    cfg = EvalConfig(claim_extractor="nonexistent")
    with pytest.raises(ValueError, match="Unknown claim_extractor"):
        _build_extractor(cfg)


# ---------------------------------------------------------------------------
# _build_judge
# ---------------------------------------------------------------------------


def test_build_judge_uses_configured_model():
    cfg = EvalConfig(judge=JudgeConfig(model="claude-3-5-sonnet-20241022"))
    judge = _build_judge(cfg)
    assert isinstance(judge, LiteLLMJudge)
    assert judge.client.model == "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# _build_metric_overrides
# ---------------------------------------------------------------------------


def test_build_metric_overrides_only_includes_requested_llm_metrics():
    cfg = EvalConfig(metrics=["coverage", "citation_number"])
    overrides = _build_metric_overrides(cfg, max_workers=1)
    assert overrides == {}


def test_build_metric_overrides_wires_precision_recall_f1():
    cfg = EvalConfig(metrics=["citation_precision", "citation_recall", "citation_f1"])
    overrides = _build_metric_overrides(cfg, max_workers=2)
    assert isinstance(overrides["citation_precision"], CitationPrecisionMetric)
    assert isinstance(overrides["citation_recall"], CitationRecallMetric)
    assert isinstance(overrides["citation_f1"], CitationF1Metric)


def test_build_metric_overrides_precision_only_still_wired():
    cfg = EvalConfig(metrics=["citation_precision"])
    overrides = _build_metric_overrides(cfg, max_workers=1)
    assert "citation_precision" in overrides
    assert "citation_recall" not in overrides
    assert "citation_f1" not in overrides
