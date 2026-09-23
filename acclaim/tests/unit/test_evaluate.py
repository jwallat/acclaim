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
    _build_filters,
    _build_judge,
    _build_metric_overrides,
    evaluate,
)
from acclaim.filters.base import ClaimFilter, FilterDecision
from acclaim.filters.check_worthiness import ClaimCategory, LLMCheckWorthinessFilter
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


# ---------------------------------------------------------------------------
# Claim filtering
# ---------------------------------------------------------------------------


class _DropContainingFilter(ClaimFilter):
    """Drops claims whose text contains *word*; records what it saw."""

    def __init__(self, word: str, name: str = "drop") -> None:
        self.word = word
        self.name = name
        self.seen: list[Claim] = []

    def decide(self, claims, answer, question=None):
        self.seen = list(claims)
        return [
            FilterDecision(
                claim=c,
                keep=self.word not in c.text,
                filter_name=self.name,
                category="TEST",
            )
            for c in claims
        ]


_FILTER_DOCS = [
    Document(doc_id="doc1", text="Paris is the capital of France."),
    Document(doc_id="doc2", text="The Eiffel Tower was completed in 1889."),
]
_FILTER_ANSWER = "Paris is the capital of France [1]. The Eiffel Tower was built in 1889 [2]."


def test_evaluate_claim_filter_override_drops_claims():
    cfg = EvalConfig(metrics=["coverage"])
    result = evaluate(
        _FILTER_ANSWER,
        _FILTER_DOCS,
        config=cfg,
        judge=_AlwaysSupportedJudge(),
        claim_filters=[_DropContainingFilter("Eiffel")],
    )
    assert len(result.steps["claims"]) == 2
    assert [c.text for c in result.steps["filtered_claims"]] == [
        "Paris is the capital of France."
    ]
    assert len(result.claims) == 1
    decisions = result.steps["filter_decisions"]
    assert result.filter_decisions is decisions
    assert [d.keep for d in decisions] == [True, False]
    assert decisions[1].category == "TEST"


def test_evaluate_multiple_filters_applied_in_order():
    first = _DropContainingFilter("Eiffel", name="first")
    second = _DropContainingFilter("nothing-matches", name="second")
    cfg = EvalConfig(metrics=["coverage"])
    result = evaluate(
        _FILTER_ANSWER,
        _FILTER_DOCS,
        config=cfg,
        judge=_AlwaysSupportedJudge(),
        claim_filters=[first, second],
    )
    # The second filter only sees what the first one kept.
    assert len(first.seen) == 2
    assert [c.text for c in second.seen] == ["Paris is the capital of France."]
    assert [d.filter_name for d in result.steps["filter_decisions"]] == [
        "first",
        "first",
        "second",
    ]


def test_evaluate_no_filters_by_default():
    cfg = EvalConfig(metrics=["coverage"])
    result = evaluate(_FILTER_ANSWER, _FILTER_DOCS, config=cfg, judge=_AlwaysSupportedJudge())
    assert result.steps["filtered_claims"] == result.steps["claims"]
    assert result.steps["filter_decisions"] == []


def test_evaluate_all_claims_filtered_out():
    cfg = EvalConfig(metrics=["coverage", "hallucination_rate"])
    result = evaluate(
        _FILTER_ANSWER,
        _FILTER_DOCS,
        config=cfg,
        judge=_AlwaysSupportedJudge(),
        claim_filters=[_DropContainingFilter("")],
    )
    assert result.claims == []
    assert result.metrics["coverage"] == pytest.approx(0.0)


def test_build_filters_empty_by_default():
    assert _build_filters(EvalConfig()) == []


def test_build_filters_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown claim filter"):
        _build_filters(EvalConfig(claim_filters=["nope"]))


def test_build_filters_check_worthiness_falls_back_to_judge():
    cfg = EvalConfig(
        claim_filters=["check_worthiness"],
        judge=JudgeConfig(model="judge-model", api_base="http://judge", max_retries=7),
    )
    cfg.check_worthiness_filter.drop_categories = ["NOT_A_CLAIM"]
    (f,) = _build_filters(cfg)
    assert isinstance(f, LLMCheckWorthinessFilter)
    assert f.client.model == "judge-model"
    assert f.client.api_base == "http://judge"
    assert f.client.max_retries == 7
    assert f.drop_categories == {ClaimCategory.NOT_A_CLAIM}


def test_build_filters_check_worthiness_own_model():
    cfg = EvalConfig(claim_filters=["check_worthiness"], judge=JudgeConfig(model="judge-model"))
    cfg.check_worthiness_filter.model = "filter-model"
    (f,) = _build_filters(cfg)
    assert f.client.model == "filter-model"
