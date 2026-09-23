"""Unit tests for batch evaluation."""

from __future__ import annotations

import threading
import time

import pytest

from acclaim.batch import BatchExample, evaluate_batch
from acclaim.config.load import ConcurrencyConfig, EvalConfig
from acclaim.data_models import Claim, Document, FilterDecision, SupportLabel, SupportResult
from acclaim.filters.base import ClaimFilter
from acclaim.judges.base import EvidenceJudge


class _AlwaysSupportedJudge(EvidenceJudge):
    def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
        return SupportResult(
            label=SupportLabel.SUPPORTED,
            confidence=1.0,
            reason="test",
        )


def test_evaluate_batch_aggregates_mean_metrics() -> None:
    cfg = EvalConfig(metrics=["coverage", "citation_number"])
    examples = [
        BatchExample(
            answer="A [1]. B [1][2].",
            documents=[
                Document(doc_id="doc-1", text="A"),
                Document(doc_id="doc-2", text="B"),
            ],
            question="What is A and B?",
        ),
        BatchExample(
            answer="C [1]. D.",
            documents=[Document(doc_id="doc-1", text="C")],
        ),
    ]

    result = evaluate_batch(examples, config=cfg, judge=_AlwaysSupportedJudge())

    assert len(result.items) == 2
    assert result.aggregate_metrics["coverage"] == pytest.approx(0.75)
    assert result.aggregate_metrics["citation_number"] == pytest.approx(2.0)

    for item, example in zip(result.items, examples):
        assert item.question == example.question
        assert item.answer == example.answer
        assert item.documents == example.documents


def test_evaluate_batch_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_batch([])


def test_evaluate_batch_parallel_caps_concurrency_at_max_workers() -> None:
    """With concurrency.max_workers > 1, evaluate_batch parallelizes across
    examples, but each example's inner per-claim judge loop must stay
    sequential — total in-flight judge calls should never exceed the
    configured max_workers (no max_workers^2 thread explosion)."""
    delay = 0.05
    max_workers = 4
    active = 0
    high_water_mark = 0
    lock = threading.Lock()

    class _TrackingJudge(EvidenceJudge):
        def evaluate(self, claim: Claim, docs: list[Document]) -> SupportResult:
            nonlocal active, high_water_mark
            with lock:
                active += 1
                high_water_mark = max(high_water_mark, active)
            time.sleep(delay)
            with lock:
                active -= 1
            return SupportResult(label=SupportLabel.SUPPORTED, confidence=1.0, reason="test")

    # 8 examples, each with 2 claims -> 16 total judge calls if fully
    # sequential, but inner loops must not add their own parallelism on top
    # of the outer batch-level parallelism.
    examples = [
        BatchExample(
            answer="A [1]. B [1].",
            documents=[Document(doc_id="doc-1", text="A and B.")],
        )
        for _ in range(8)
    ]
    cfg = EvalConfig(
        metrics=["coverage"],
        relevance_stratified_metrics=[],
        concurrency=ConcurrencyConfig(max_workers=max_workers),
    )

    start = time.perf_counter()
    result = evaluate_batch(examples, config=cfg, judge=_TrackingJudge())
    elapsed = time.perf_counter() - start

    assert len(result.items) == 8
    assert high_water_mark <= max_workers
    # 16 sequential calls would take ~16*delay; parallel across max_workers
    # examples should be meaningfully faster.
    assert elapsed < (16 * delay) / 2


class _DropWordFilter(ClaimFilter):
    name = "drop_word"

    def __init__(self, word: str) -> None:
        self.word = word

    def decide(self, claims, answer, question=None):
        return [
            FilterDecision(
                claim=c,
                keep=self.word not in c.text,
                filter_name=self.name,
                category="DROPPED" if self.word in c.text else "KEPT",
            )
            for c in claims
        ]


def test_evaluate_batch_claim_filters_and_summary() -> None:
    docs = [Document(doc_id="d1", text="A."), Document(doc_id="d2", text="B.")]
    examples = [
        BatchExample(answer="Keep this [1]. Drop this [2].", documents=docs),
        BatchExample(answer="Drop this [1].", documents=docs),
    ]
    result = evaluate_batch(
        examples,
        config=EvalConfig(metrics=["coverage"]),
        judge=_AlwaysSupportedJudge(),
        claim_filters=[_DropWordFilter("Drop")],
    )
    assert [len(item.claims) for item in result.items] == [1, 0]
    assert len(result.items[0].filter_decisions) == 2
    s = result.filter_summary
    assert s["claims_extracted"] == 3
    assert s["claims_kept"] == 1
    assert s["claims_dropped"] == 2
    assert s["drop_rate"] == pytest.approx(2 / 3)
    assert s["items_with_drops"] == 2
    assert s["items_all_dropped"] == 1
    assert s["by_filter"]["drop_word"]["categories"] == {
        "KEPT": {"kept": 1, "dropped": 0},
        "DROPPED": {"kept": 0, "dropped": 2},
    }


def test_evaluate_batch_no_filter_summary_without_filters() -> None:
    docs = [Document(doc_id="d1", text="A.")]
    result = evaluate_batch(
        [BatchExample(answer="Fact [1].", documents=docs)],
        config=EvalConfig(metrics=["coverage"]),
        judge=_AlwaysSupportedJudge(),
    )
    assert result.filter_summary is None
